from cmath import cos
import random
import lightgbm
import numpy as np

from gensim.models import TfidfModel
from gensim.models import LsiModel
from gensim.corpora import Dictionary
from scipy.spatial.distance import cosine

NUM_NEGATIVES = 1
NUM_LATENT_TOPICS = 200

class LETOR:
    def __init__(self):
        self.documents = {}
        self.queries = {}

        with open("nfcorpus/train.docs") as file:
            for line in file:
                doc_id, content = line.split("\t")
                self.documents[doc_id] = content.split()

        with open("nfcorpus/train.vid-desc.queries", encoding='utf-8') as file:
            for line in file:
                q_id, content = line.split("\t")
                self.queries[q_id] = content.split()
        
    def training_prep(self):
        #relevance level: 3 (fullt relevant), 2 (partially relevant), 1 (marginally relevant)
        #
        #grouping by q_id   
        self.q_docs_rel = {}
        with open("nfcorpus/train.3-2-1.qrel") as file:
            for line in file:
                q_id, _, doc_id, rel = line.split("\t")
                if (q_id in self.queries) and (doc_id in self.documents):
                    if q_id not in self.q_docs_rel:
                        self.q_docs_rel[q_id] = []
                    self.q_docs_rel[q_id].append((doc_id, int(rel)))

        #group_qid_count untuk model LGBMRanker
        self.group_qid_count = []
        self.dataset = []
        for q_id in self.q_docs_rel:
            docs_rels = self.q_docs_rel[q_id]
            self.group_qid_count.append(len(docs_rels) + NUM_NEGATIVES)
            for doc_id, rel in docs_rels:
                self.dataset.append((self.queries[q_id], self.documents[doc_id], rel))
            #tambahkan satu negative
            self.dataset.append((self.queries[q_id], random.choice(list(self.documents.values())), 0))
        

    def building_model(self):
        self.dictionary = Dictionary()
        bow_corpus = [self.dictionary.doc2bow(doc, allow_update = True) for doc in self.documents.values()]
        self.model = LsiModel(bow_corpus, num_topics = NUM_LATENT_TOPICS) # 200 latent topics
        #print(self.vector_rep(self.documents["MED-329"])) gatau kenapa masih beda

    def vector_rep(self, text):
        rep = [topic_value for (_, topic_value) in self.model[self.dictionary.doc2bow(text)]]
        return rep if len(rep) == NUM_LATENT_TOPICS else [0.] * NUM_LATENT_TOPICS

    def features(self, query, doc):
        v_q = self.vector_rep(query)
        v_d = self.vector_rep(doc)
        q = set(query)
        d = set(doc)
        cosine_dist = cosine(v_q, v_d)
        jaccard = len(q & d) / len(q | d)
        return v_q + v_d + [jaccard] + [cosine_dist]
    
    def separate_dataset(self):
        self.X = []
        self.Y = []
        for (query, doc, rel) in self.dataset:
            self.X.append(self.features(query, doc))
            self.Y.append(rel)
        
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)

    def training_ranker(self):
        self.ranker = lightgbm.LGBMRanker(
                            objective="lambdarank",
                            boosting_type = "gbdt",
                            n_estimators = 100,
                            importance_type = "gain",
                            metric = "ndcg",
                            num_leaves = 40,
                            learning_rate = 0.02,
                            max_depth = -1)

        self.ranker.fit(self.X, self.Y,
                group = self.group_qid_count,
                verbose = 10)

    def training(self):
        #training preparations
        self.training_prep()

        #building LSI/LSA Model
        self.building_model()
        self.separate_dataset()

        #training the ranker
        self.training_ranker()

    def predict(self, query, docs):
        X_unseen = []

        for doc_id, doc in docs:
            X_unseen.append(self.features(query.split(), doc.split()))
        X_unseen = np.array(X_unseen)
        scores = self.ranker.predict(X_unseen)

        did_scores = [x for x in zip([did for (did, _) in docs], scores)]
        sorted_did_scores = sorted(did_scores, key = lambda tup: tup[1], reverse = True)

        return sorted_did_scores

if __name__ == '__main__':

    query = "how much cancer risk can be avoided through lifestyle change ?"

    docs = [("D1", "dietary restriction reduces insulin-like growth factor levels modulates apoptosis cell proliferation tumor progression num defici pubmed ncbi abstract diet contributes one-third cancer deaths western world factors diet influence cancer elucidated reduction caloric intake dramatically slows cancer progression rodents major contribution dietary effects cancer insulin-like growth factor igf-i lowered dietary restriction dr humans rats igf-i modulates cell proliferation apoptosis tumorigenesis mechanisms protective effects dr depend reduction multifaceted growth factor test hypothesis igf-i restored dr ascertain lowering igf-i central slowing bladder cancer progression dr heterozygous num deficient mice received bladder carcinogen p-cresidine induce preneoplasia confirmation bladder urothelial preneoplasia mice divided groups ad libitum num dr num dr igf-i igf-i/dr serum igf-i lowered num dr completely restored igf-i/dr-treated mice recombinant igf-i administered osmotic minipumps tumor progression decreased dr restoration igf-i serum levels dr-treated mice increased stage cancers igf-i modulated tumor progression independent body weight rates apoptosis preneoplastic lesions num times higher dr-treated mice compared igf/dr ad libitum-treated mice administration igf-i dr-treated mice stimulated cell proliferation num fold hyperplastic foci conclusion dr lowered igf-i levels favoring apoptosis cell proliferation ultimately slowing tumor progression mechanistic study demonstrating igf-i supplementation abrogates protective effect dr neoplastic progression"), 
       ("D2", "study hard as your blood boils"), 
       ("D3", "processed meats risk childhood leukemia california usa pubmed ncbi abstract relation intake food items thought precursors inhibitors n-nitroso compounds noc risk leukemia investigated case-control study children birth age num years los angeles county california united states cases ascertained population-based tumor registry num num controls drawn friends random-digit dialing interviews obtained num cases num controls food items principal interest breakfast meats bacon sausage ham luncheon meats salami pastrami lunch meat corned beef bologna hot dogs oranges orange juice grapefruit grapefruit juice asked intake apples apple juice regular charcoal broiled meats milk coffee coke cola drinks usual consumption frequencies determined parents child risks adjusted risk factors persistent significant associations children's intake hot dogs odds ratio num num percent confidence interval ci num num num hot dogs month trend num fathers intake hot dogs num ci num num highest intake category trend num evidence fruit intake provided protection results compatible experimental animal literature hypothesis human noc intake leukemia risk potential biases data study hypothesis focused comprehensive epidemiologic studies warranted"), 
       ("D4", "long-term effects calorie protein restriction serum igf num igfbp num concentration humans summary reduced function mutations insulin/igf-i signaling pathway increase maximal lifespan health span species calorie restriction cr decreases serum igf num concentration num protects cancer slows aging rodents long-term effects cr adequate nutrition circulating igf num levels humans unknown report data long-term cr studies num num years showing severe cr malnutrition change igf num igf num igfbp num ratio levels humans contrast total free igf num concentrations significantly lower moderately protein-restricted individuals reducing protein intake average num kg num body weight day num kg num body weight day num weeks volunteers practicing cr resulted reduction serum igf num num ng ml num num ng ml num findings demonstrate unlike rodents long-term severe cr reduce serum igf num concentration igf num igfbp num ratio humans addition data provide evidence protein intake key determinant circulating igf num levels humans suggest reduced protein intake important component anticancer anti-aging dietary interventions"), 
       ("D5", "cancer preventable disease requires major lifestyle abstract year num million americans num million people worldwide expected diagnosed cancer disease commonly believed preventable num num cancer cases attributed genetic defects remaining num num roots environment lifestyle lifestyle factors include cigarette smoking diet fried foods red meat alcohol sun exposure environmental pollutants infections stress obesity physical inactivity evidence cancer-related deaths num num due tobacco num num linked diet num num due infections remaining percentage due factors radiation stress physical activity environmental pollutants cancer prevention requires smoking cessation increased ingestion fruits vegetables moderate alcohol caloric restriction exercise avoidance direct exposure sunlight minimal meat consumption grains vaccinations regular check-ups review present evidence inflammation link agents/factors cancer agents prevent addition provide evidence cancer preventable disease requires major lifestyle")]
    
    LETOR_instance = LETOR()
    sorted_did_scores = LETOR_instance.predict(query, docs)

    assert LETOR_instance.documents["MED-329"] == ['phosphate', 'vascular', 'toxin', 'pubmed', 'ncbi', 'abstract', 'elevated', 'phosphate', 'levels', 'advanced', 'renal', 'failure', 'dysregulated', 'calcium', 'parathyroid', 'hormone', 'vitamin', 'levels', 'contribute', 'complex', 'chronic', 'kidney', 'disease-mineral', 'bone', 'disease', 'ckd-mbd', 'converging', 'evidence', 'vitro', 'clinical', 'epidemiological', 'studies', 'suggest', 'increased', 'vascular', 'calcification', 'mortality', 'vessels', 'exposed', 'high', 'conditions', 'vitro', 'develop', 'apoptosis', 'convert', 'bone-like', 'cells', 'develop', 'extensive', 'calcification', 'clinical', 'studies', 'children', 'dialysis', 'show', 'high', 'increased', 'vessel', 'wall', 'thickness', 'arterial', 'stiffness', 'coronary', 'calcification', 'epidemiological', 'studies', 'adult', 'dialysis', 'patients', 'demonstrate', 'significant', 'independent', 'association', 'raised', 'mortality', 'importantly', 'raised', 'cardiovascular', 'pre-dialysis', 'ckd', 'subjects', 'normal', 'renal', 'function', 'high', 'binders', 'effectively', 'reduce', 'serum', 'decrease', 'linked', 'improved', 'survival', 'raised', 'serum', 'triggers', 'release', 'fibroblast', 'growth', 'factor', 'num', 'fgf', 'num', 'beneficial', 'effect', 'increasing', 'excretion', 'early', 'ckd', 'increased', 'num', 'fold', 'dialysis', 'independent', 'cardiovascular', 'risk', 'factor', 'fgf', 'num', 'co-receptor', 'klotho', 'direct', 'effects', 'vasculature', 'leading', 'calcification', 'fascinatingly', 'disturbances', 'fgf', 'num', 'klotho', 'raised', 'premature', 'aging', 'data', 'suggest', 'high', 'levels', 'adverse', 'vascular', 'effects', 'maintaining', 'serum', 'levels', 'normal', 'range', 'reduces', 'cardiovascular', 'risk', 'mortality'], "isi doc1 salah"
    assert LETOR_instance.documents["MED-330"] == ['dietary', 'phosphorus', 'acutely', 'impairs', 'endothelial', 'function', 'abstract', 'excessive', 'dietary', 'phosphorus', 'increase', 'cardiovascular', 'risk', 'healthy', 'individuals', 'patients', 'chronic', 'kidney', 'disease', 'mechanisms', 'underlying', 'risk', 'completely', 'understood', 'determine', 'postprandial', 'hyperphosphatemia', 'promote', 'endothelial', 'dysfunction', 'investigated', 'acute', 'effect', 'phosphorus', 'loading', 'endothelial', 'function', 'vitro', 'vivo', 'exposing', 'bovine', 'aortic', 'endothelial', 'cells', 'phosphorus', 'load', 'increased', 'production', 'reactive', 'oxygen', 'species', 'depended', 'phosphorus', 'influx', 'sodium-dependent', 'phosphate', 'transporters', 'decreased', 'nitric', 'oxide', 'production', 'inhibitory', 'phosphorylation', 'endothelial', 'nitric', 'oxide', 'synthase', 'phosphorus', 'loading', 'inhibited', 'endothelium-dependent', 'vasodilation', 'rat', 'aortic', 'rings', 'num', 'healthy', 'men', 'alternately', 'served', 'meals', 'num', 'mg', 'num', 'mg', 'phosphorus', 'double-blind', 'crossover', 'study', 'measured', 'flow-mediated', 'dilation', 'brachial', 'artery', 'num', 'meals', 'high', 'dietary', 'phosphorus', 'load', 'increased', 'serum', 'phosphorus', 'num', 'significantly', 'decreased', 'flow-mediated', 'dilation', 'flow-mediated', 'dilation', 'correlated', 'inversely', 'serum', 'phosphorus', 'findings', 'suggest', 'endothelial', 'dysfunction', 'mediated', 'acute', 'postprandial', 'hyperphosphatemia', 'contribute', 'relationship', 'serum', 'phosphorus', 'level', 'risk', 'cardiovascular', 'morbidity', 'mortality'], "isi doc2 salah"

    assert LETOR_instance.queries["PLAIN-2428"] == ['the', 'parable', 'of', 'the', 'tiny', 'parachute', 'explains', 'the', 'study', 'that', 'found', 'no', 'relationship', 'between', 'dietary', 'fiber', 'intake', 'and', 'diverticulosis', '.'], "isi query1 salah"
    assert LETOR_instance.queries["PLAIN-2435"] == ['most', 'people', 'have', 'between', '3', 'bowel', 'movements', 'a', 'day', 'and', '3', 'a', 'week', ',', 'but', 'normal', 'doesn', '’', 't', 'necessarily', 'mean', 'optimal', '.'], "isi query2 salah"

    assert len(LETOR_instance.dataset) == 28277, "panjang dataset salah"
    assert LETOR_instance.group_qid_count == [102, 16, 48, 68, 37, 17, 32, 37, 28, 64, 22, 35, 74, 73, 66, 30, 46, 48, 62, 71, 58, 90, 63, 43, 60, 51, 71, 54, 32, 40, 36, 44, 85, 38, 35, 83, 62, 56, 37, 42, 49, 96, 41, 32, 14, 50, 22, 14, 18, 26, 65, 25, 41, 81, 42, 39, 58, 23, 110, 25, 37, 67, 62, 44, 53, 51, 15, 42, 52, 76, 20, 52, 103, 53, 58, 25, 34, 19, 33, 25, 36, 69, 106, 58, 82, 17, 48, 92, 72, 18, 65, 94, 82, 36, 42, 52, 39, 24, 42, 45, 50, 50, 59, 8, 36, 66, 28, 50, 48, 44, 22, 90, 25, 56, 59, 39, 34, 39, 55, 34, 20, 21, 44, 29, 31, 51, 39, 105, 52, 57, 107, 75, 45, 40, 72, 45, 53, 53, 50, 120, 38, 65, 84, 48, 50, 66, 21, 61, 27, 38, 84, 66, 42, 11, 46, 68, 16, 33, 85, 50, 65, 50, 60, 21, 41, 29, 53, 92, 12, 16, 63, 42, 48, 43, 5, 45, 21, 47, 35, 31, 40, 26, 43, 18, 35, 21, 77, 10, 79, 45, 25, 47, 24, 38, 11, 34, 26, 32, 25, 12, 13, 44, 22, 29, 26, 45, 23, 22, 24, 38, 49, 18, 42, 64, 26, 49, 80, 84, 56, 50, 83, 44, 65, 13, 43, 29, 20, 28, 71, 82, 43, 84, 42, 50, 24, 56, 41, 24, 28, 52, 41, 60, 45, 43, 40, 37, 33, 44, 80, 44, 41, 47, 49, 52, 34, 33, 58, 80, 61, 33, 44, 55, 77, 24, 25, 16, 42, 43, 61, 29, 27, 20, 32, 56, 34, 69, 20, 34, 46, 34, 43, 18, 33, 43, 30, 10, 26, 54, 65, 48, 30, 19, 17, 13, 28, 16, 24, 58, 29, 28, 31, 39, 26, 13, 41, 31, 4, 11, 39, 22, 24, 33, 28, 27, 40, 18, 36, 64, 36, 24, 90, 24, 6, 12, 4, 5, 31, 9, 4, 7, 108, 15, 14, 16, 3, 26, 10, 14, 17, 28, 9, 19, 21, 5, 70, 7, 8, 26, 7, 38, 23, 19, 21, 24, 39, 7, 14, 14, 18, 31, 12, 130, 23, 7, 33, 30, 28, 32, 49, 10, 84, 58, 37, 35, 37, 40, 29, 33, 65, 12, 25, 26, 2, 11, 4, 82, 12, 357, 49, 29, 54, 3, 31, 31, 27, 12, 20, 17, 22, 214, 38, 12, 85, 41, 18, 26, 7, 22, 47, 32, 35, 14, 11, 8, 23, 20, 21, 20, 14, 29, 20, 26, 5, 12, 15, 4, 15, 9, 13, 12, 43, 9, 11, 22, 4, 11, 12, 10, 3, 5, 10, 11, 11, 2, 25, 28, 4, 2, 2, 9, 36, 6, 2, 5, 37, 2, 3, 3, 3, 6, 10, 13, 5, 3, 2, 4, 3, 14, 7, 5, 4, 2, 2, 22, 4, 26, 2, 10, 6, 12, 3, 5, 8, 5, 3, 4, 3, 5, 15, 3, 4, 2, 6, 8, 2, 7, 5, 8, 6, 3, 15, 8, 2, 24, 6, 2, 24, 29, 75, 37, 271, 4, 6, 43, 2, 2, 4, 15, 4, 3, 6, 4, 3, 25, 5, 25, 3, 4, 5, 44, 19, 2, 5, 2, 29, 23, 54, 25, 32, 25, 11, 21, 45, 48, 32, 60, 21, 19, 64, 67, 55, 65, 39, 58, 19, 31, 52, 37, 51, 14, 17, 20, 37, 22, 15, 18, 18, 43, 43, 42, 45, 11, 28, 5, 17, 36, 56, 19, 19, 14, 52, 7, 36, 54, 15, 25, 22, 55, 42, 11, 34, 15, 63, 6, 36, 48, 62, 6, 27, 2, 7, 9, 23, 16, 18, 14, 33, 17, 53, 248, 12, 47, 37, 17, 62, 23, 47, 51, 53, 153, 18, 139, 32, 52, 16, 10, 36, 33, 37, 11, 4, 23, 41, 15, 269, 38, 42, 34, 24, 21, 3, 21, 57, 27, 29, 17, 18, 10, 56, 6, 29, 29, 32, 35, 61, 63, 7, 115, 7, 6, 18, 4, 6, 34, 30, 10, 2, 180, 12, 32, 36, 41, 9, 15, 40, 57, 54, 32, 7, 15, 20, 19, 12, 24, 26, 9, 24, 68, 29, 5, 16, 16, 45, 63, 99, 37, 44, 34, 38, 19, 10, 24, 5, 77, 42, 22, 13, 12, 8, 20, 59, 2, 9, 17, 19, 2, 90, 15, 12, 51, 36, 9, 50, 33, 34, 9, 19, 13, 7, 35, 35, 15, 13, 3, 2, 26, 4, 39, 55, 23, 139, 33, 30, 27, 23, 48, 16, 19, 20, 26, 31, 26, 27, 13, 20, 19, 23, 4, 17, 33, 58, 54, 34, 112, 34, 20, 3, 14, 13, 52, 10, 19, 3, 3, 10, 152, 14, 20, 35, 13, 47, 8, 39, 19, 20, 66, 40, 9, 27, 15, 31, 8, 27, 29, 44, 90, 78, 102, 39, 53, 44, 120, 34, 18, 82, 32, 29, 22, 22, 22, 60, 84], "group_qid_count salah"
    assert sum(LETOR_instance.group_qid_count) == len(LETOR_instance.dataset), "ada yang salah"
    assert LETOR_instance.dataset[:2] == [(['diet', 'and', 'exercise', 'synergize', 'to', 'improve', 'endothelial', 'function', ',', 'the', 'ability', 'of', 'our', 'arteries', 'to', 'relax', 'normally', '.'], ['curcumin', 'ingestion', 'exercise', 'training', 'improve', 'vascular', 'endothelial', 'function', 'postmenopausal', 'women', 'pubmed', 'ncbi', 'abstract', 'vascular', 'endothelial', 'function', 'declines', 'aging', 'increased', 'risk', 'cardiovascular', 'disease', 'lifestyle', 'modification', 'aerobic', 'exercise', 'dietary', 'adjustment', 'favorable', 'effect', 'vascular', 'aging', 'curcumin', 'major', 'component', 'turmeric', 'anti-inflammatory', 'anti-oxidative', 'effects', 'investigated', 'effects', 'curcumin', 'ingestion', 'aerobic', 'exercise', 'training', 'flow-mediated', 'dilation', 'indicator', 'endothelial', 'function', 'postmenopausal', 'women', 'total', 'num', 'postmenopausal', 'women', 'assigned', 'num', 'groups', 'control', 'exercise', 'curcumin', 'groups', 'curcumin', 'group', 'ingested', 'curcumin', 'orally', 'num', 'weeks', 'exercise', 'group', 'underwent', 'moderate', 'aerobic', 'exercise', 'training', 'num', 'weeks', 'intervention', 'flow-mediated', 'dilation', 'measured', 'difference', 'baseline', 'flow-mediated', 'dilation', 'key', 'dependent', 'variables', 'detected', 'groups', 'flow-mediated', 'dilation', 'increased', 'significantly', 'equally', 'curcumin', 'exercise', 'groups', 'observed', 'control', 'group', 'results', 'curcumin', 'ingestion', 'aerobic', 'exercise', 'training', 'increase', 'flow-mediated', 'dilation', 'postmenopausal', 'women', 'suggesting', 'potentially', 'improve', 'age-related', 'decline', 'endothelial', 'function', 'copyright', 'num', 'elsevier', 'rights', 'reserved'], 3), (['diet', 'and', 'exercise', 'synergize', 'to', 'improve', 'endothelial', 'function', ',', 'the', 'ability', 'of', 'our', 'arteries', 'to', 'relax', 'normally', '.'], ['endothelial', 'dysfunction', 'early', 'predictor', 'atherosclerosis', 'abstract', 'abstract', 'discovery', 'num', 'nitric', 'oxide', 'fact', 'elusive', 'endothelium-derived', 'relaxing', 'factor', 'evident', 'major', 'cardiovascular', 'signalling', 'molecule', 'bioavailability', 'crucial', 'determining', 'atherosclerosis', 'develop', 'sustained', 'high', 'levels', 'harmful', 'circulating', 'stimuli', 'cardiovascular', 'risk', 'factors', 'diabetes', 'mellitus', 'elicit', 'responses', 'endothelial', 'cells', 'sequentially', 'endothelial', 'cell', 'activation', 'endothelial', 'dysfunction', 'ed', 'ed', 'characterised', 'reduced', 'bioavailability', 'recognised', 'early', 'reversible', 'precursor', 'atherosclerosis', 'pathogenesis', 'ed', 'multifactorial', 'oxidative', 'stress', 'appears', 'common', 'underlying', 'cellular', 'mechanism', 'ensuing', 'loss', 'vaso-active', 'inflammatory', 'haemostatic', 'redox', 'homeostasis', 'body', 'vascular', 'system', 'role', 'ed', 'pathophysiological', 'link', 'early', 'endothelial', 'cell', 'cardiovascular', 'risk', 'factors', 'development', 'ischaemic', 'heart', 'disease', 'importance', 'basic', 'scientists', 'clinicians', 'alike'], 3)], "isi dataset salah"

    #assert LETOR_instance.vector_rep(LETOR_instance.documents["MED-329"]) == [5.9207874649414505, -3.1358543082589088, -2.6077567003839732, 0.6843319142545866, -1.8058665544513566, -2.381580532138159, -1.4732444657643051, -0.27873127872227144, 0.5551310779832449, -1.3395383197884914, -0.7153786683992718, 0.4581364420247896, -1.6566098843650232, -0.2913277536833165, -1.7412154350070537, 0.7543792588118651, 0.7469511741255683, 1.0829986220643602, -1.063502185815146, -1.0675383991453011, -0.1631987326710983, 1.7016046961678888, -1.4945021328851116, -1.514737664397233, -0.5818601019597628, 1.562891894908965, 0.9544145993805866, 0.027280683764285833, 0.12542366495340607, 0.4233737586937925, -0.10745374857244488, -1.5077519876868772, -0.4904822510102606, -1.4728540784057986, 1.2156064339537072, -0.1407798072981922, 0.9517791544423931, -1.9355748215425737, -1.2736212604711388, 0.3394980986622505, -1.533577326385445, -0.03246346472893277, 0.08378221760756216, -0.4696648243618851, 0.3525599514475394, -1.2609080061958933, 1.0128613610507977, 0.759220720711047, 0.8912536899326289, -0.08769296477502139, -1.3100826597708266, -0.8839703174683985, -0.2908971166770412, 1.1325324409718842, 0.7631954516322188, 0.16370341855671725, -0.7535708283968217, -0.5581597577271532, -0.4316893097441788, 0.8082273027565944, 0.20546582644589906, 0.15719709881566152, -0.6709296023090464, 1.0220777164454422, 0.4068200588768518, 1.2973059309400208, 0.539369538466261, -0.06254888737099094, -1.005876894946242, 0.18392659605118925, 1.2704886931971167, -0.6094255186573919, 0.7410687375381098, 0.8532102554584138, -0.20793948446062893, -0.04101938544734737, -0.08413120175060751, -0.9060399848497839, -1.107008035643804, 0.768343106556213, 0.01728505170099294, 0.0956072405792872, -0.09688008179211352, -1.0369522536539755, -0.32749007536328323, 0.32572989492296356, -0.8036713839052181, -0.1899633763358968, 0.7077509320024515, 0.5301481911498879, 0.27726610208557206, -0.6764643801102381, -0.1289407612295924, 0.5599983605500164, 0.825922513480161, 0.04314951853950428, 0.5106109450282678, -0.4396059726535655, 0.6716583579491835, 0.41389266072545605, -0.7800233834669986, 0.0429904659518326, -0.5881711427578712, 0.17617801382075332, 0.2473580346976986, 0.06744758128648669, -0.19305953231918602, 0.12707137503931284, -0.02698857429184682, 0.13198911118662685, 0.5679795899752977, 0.6954428590765424, -0.1394282007148834, -0.1058301160099721, 0.7261715526071805, -0.7228792115172135, -0.28037199275228125, -0.2332686821875333, 0.902031736575649, -0.6598051723678626, -0.3019181371425208, -0.1398511928326017, -0.32133592205220085, 0.5103175549681948, 0.6197939653305208, 0.016520456268257596, -0.670654902810984, 0.11615737770399821, 0.05673109861087061, -0.32652844532697195, 0.5863347390212258, 0.17383857631102662, 0.05874695867744955, 0.17690650408503308, -0.2826503138707112, 0.4157515492501903, 0.3283358771422104, -0.505611761501378, 0.27313435472777786, 0.619428796872319, -0.16458867374352196, -0.5694155639924066, 0.09632238398621389, 0.5540398389493093, 0.3966988994490467, -0.251905575712161, 0.43748116328071246, -0.7339223841211493, -0.11405242237710043, 0.147387927649596, -0.1006268006425673, -0.3840446899186011, 0.09243113009593315, -0.2793525567796369, 0.512870514206953, 0.4250776252142135, 0.31116280845062305, 0.22492836874742472, -0.31672677256799386, -0.20287923116342432, -0.35326170679279, 0.10424983856529478, -0.29174663476580676, -0.02362221564326705, -0.2639048586193841, -0.024883353883107414, -0.23633529146031906, 0.23431684219934523, -0.41279989737687867, 0.5619282060994424, 0.16611824619614413, 0.17769576264098216, 0.35150707368390677, -0.5650641307507925, 0.43845842692178044, -0.11203155320162148, -0.04797774128634831, -0.12427370870761041, 0.18510037544231514, 0.143745653963239, -0.13695229934086933, 0.19484185240266816, -0.26691566789804805, -0.23171754377124262, 0.16980967309272477, -0.34266383988506444, -0.3759735077137762, -0.44375092752453377, 0.17083006900065836, 0.05392575796193215, 0.26239257624884066, -0.16936734944805476, -0.2689656419545478, -0.05685001324476788, -0.19062560519022234, 0.18136322453503245, 0.5032179020451062, 0.27197204362411337, 0.37203664130551484, 0.1965894242591994], "vector rep doc salah"
    #assert LETOR_instance.vector_rep(LETOR_instance.queries["PLAIN-2435"]) == [0.02544844471796418, -0.042704070767094496, -0.1252775639347356, 0.04297510275135637, -0.03913016367942445, 0.03191360073847664, 0.03567569569616808, -0.05248464597604366, 0.012795067104028932, -0.019032105692463226, -0.0535681290886673, 0.016688047989089017, 0.05887419741158889, 0.02813069712771045, 0.009979343006109368, 0.03753954394545956, -0.010963851795696516, -0.00479277179305932, -0.10947532077885162, -0.05662464235881432, -0.03432322086763974, -0.11472220240105843, 0.08689538599420085, 0.02643287160596198, 0.012834572625523913, -0.05662336715693626, 0.12126562143232356, 0.090617893046512, -0.03653481536247804, -0.08650785214715978, 0.060628337197547254, 0.003719018408923478, 0.06723901733770006, -0.01626581048419388, 0.07209471462381373, -0.21382396409550725, 0.047888005088906824, 0.04321725761462204, -0.13228910644250824, -0.0012966995178779418, -0.0030298902151286353, 0.07706793428031596, 0.0704417375540581, 0.05029905971149381, 0.00393009693576734, 0.0012708393781523165, -0.06156016899850555, 0.10493278639677076, 0.05636171111484129, 0.06140881437338466, -0.030844908934767806, 0.09066920290043637, 0.06796560363229169, -0.10970058123163816, -0.12952326346587925, 0.051605601816216644, -0.0234276469920314, 0.09485844247596434, -0.01739558442627928, 0.04555795685527807, -0.05506962498106566, 0.017408753066057308, -0.02143617461564886, 0.10820725515587026, 0.21723339362110036, 0.018185405304019537, -0.059990678450072873, -0.007279433446433088, -0.002493829015848699, 0.06798195234415479, -0.04746891709934635, 0.18807877374773557, -0.09714035795459364, 0.003148238840537813, 0.10205382947589693, 0.13426332790833917, 0.0025892840097922476, -0.028710545916038215, -0.0985653066639281, -0.005550667566912786, -0.0422169009539172, -0.06207537672024664, 0.15519583308542348, -0.15800271778974595, 0.01921838325017458, -0.03646215723417601, 0.023178226382218208, -0.004839893027878629, 0.04681731935656125, -0.015495976084135086, -0.03679585833550739, 0.009819376688376784, -0.06181128288595802, -0.26445747535824576, -0.07264156187926607, 0.09524958075216425, -0.045900018518101104, 0.0697031218502219, 0.0544958508160274, -0.05393239843679719, 0.037749833851676726, -0.008782666709510455, -0.060979142850841744, 0.1631667534138683, 0.16448573777482753, -0.2629352109546658, 0.06845067041825954, 0.015434448006592064, 0.045656490226890264, -0.17037123521803768, -0.13937517384741427, 0.1478167168742218, 0.039391967360920044, -0.035819757411610066, 0.14240099645625542, 0.017085316932038134, 0.13205467905116927, 0.014999518321672555, 0.04289020440281257, 0.0014620811649157066, 0.01291857536143613, 0.03719850146653221, 0.08434402025862008, 0.23498669058389668, 0.0003359500771395922, -0.16818555080466516, -0.028733006328129168, -0.18040811719959193, 0.1204623060441145, 0.19067695668373766, 8.1966875354415e-05, 0.03584518268583916, 0.1611369072960347, -0.03162279020164392, -0.13069146216022257, 0.10023256069626543, 0.11000243458554632, -0.17614064407388172, 0.019207523242046737, -0.038992235744310905, 0.06057058160161411, -0.05111308183346618, -0.03742252509097745, -0.0976823219180053, -0.06356220704567202, 0.07920766743370178, 0.015056985087080726, -0.25149013363286726, 0.011319990293475926, -0.1470184283658753, 0.001180562267021739, -0.018548427555530086, 0.12300478855751946, -0.13342529053720956, 0.020272601955610276, -0.1963399167912313, -0.0812880479090092, -0.17265336700074208, 0.07309585559529293, -0.016390778359675305, 0.17989660327192636, 0.002088614961246372, 0.07629814965375338, -0.05144287326457018, -0.019909482771404252, -0.013507484956837314, -0.07094791849091177, 0.0022530270712047013, -0.06849895438859784, -0.07076536210707628, 0.008202192123303684, -0.1560863243389607, 0.07052076098195278, 0.15172145476573926, -0.017908987311933536, -0.06567671131182204, 0.04759127762044353, -0.04213831261641438, 0.12650859146563265, -0.19897046267367216, 0.05373855948435799, -0.11636161477159217, -0.1021559552405937, -0.003252978193537159, -0.12885639417221692, -0.06181830412901282, -0.18538580156116485, -0.011003111217888273, 0.06983483953167946, -0.09211824725821394, 0.17386736443515394, 0.07159376535086528, -0.00159472591120746, -0.019830459583203643, -0.04218090835672526, 0.13489183127834187, 0.009815044435175961, -0.05298078062729526, -0.08605662022820217, -0.07637894058353442], "vector rep query salah"

    print("query        :", query)
    print("SERP/Ranking :")
    for (did, score) in sorted_did_scores:
        print(did, score)


    




