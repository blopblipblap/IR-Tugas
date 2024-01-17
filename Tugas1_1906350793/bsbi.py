from operator import itemgetter
import os
import pickle
import contextlib
import heapq
import time

from sqlalchemy import Integer

from index import InvertedIndexReader, InvertedIndexWriter
from util import IdMap, sorted_intersect
from compression import StandardPostings, VBEPostings
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from tqdm import tqdm

class BSBIIndex:
    """
    Attributes
    ----------
    term_id_map(IdMap): Untuk mapping terms ke termIDs
    doc_id_map(IdMap): Untuk mapping relative paths dari dokumen (misal,
                    /collection/0/gamma.txt) to docIDs
    data_dir(str): Path ke data
    output_dir(str): Path ke output index files
    postings_encoding: Lihat di compression.py, kandidatnya adalah StandardPostings,
                    VBEPostings, dsb.
    index_name(str): Nama dari file yang berisi inverted index
    """
    def __init__(self, data_dir, output_dir, postings_encoding, index_name = "main_index"):
        self.term_id_map = IdMap()
        self.doc_id_map = IdMap()
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.index_name = index_name
        self.postings_encoding = postings_encoding

        # Untuk menyimpan nama-nama file dari semua intermediate inverted index
        self.intermediate_indices = []

    def save(self):
        """Menyimpan doc_id_map and term_id_map ke output directory via pickle"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'wb') as f:
            pickle.dump(self.term_id_map, f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'wb') as f:
            pickle.dump(self.doc_id_map, f)

    def load(self):
        """Memuat doc_id_map and term_id_map dari output directory"""

        with open(os.path.join(self.output_dir, 'terms.dict'), 'rb') as f:
            self.term_id_map = pickle.load(f)
        with open(os.path.join(self.output_dir, 'docs.dict'), 'rb') as f:
            self.doc_id_map = pickle.load(f)

    def parse_block(self, block_dir_relative):
        """
        Lakukan parsing terhadap text file sehingga menjadi sequence of
        <termID, docID> pairs.

        Gunakan tools available untuk Stemming Bahasa Indonesia Seperti
        PySastrawi:  https://github.com/har07/PySastrawi

        JANGAN LUPA BUANG STOPWORDS! di PySastrawi juga ada daftar Indonesian
        stopwords.

        Untuk "sentence segmentation" dan "tokenization", bisa menggunakan
        regex atau boleh juga menggunakan tools lain yang berbasis machine
        learning.

        Parameters
        ----------
        block_dir_relative : str
            Relative Path ke directory yang mengandung text files untuk sebuah block.

            CATAT bahwa satu folder di collection dianggap merepresentasikan satu block.
            Konsep block di soal tugas ini berbeda dengan konsep block yang terkait
            dengan operating systems.

        Returns
        -------
        List[Tuple[Int, Int]]
            Returns all the td_pairs extracted from the block
            Mengembalikan semua pasangan <termID, docID> dari sebuah block (dalam hal
            ini sebuah sub-direktori di dalam folder collection)

        Harus menggunakan self.term_id_map dan self.doc_id_map untuk mendapatkan
        termIDs dan docIDs. Dua variable ini harus persis untuk semua pemanggilan
        parse_block(...).
        """
        """
        alur dulu bang
        1. stop word -> ada di pysastrawi
        2. stemming -> ada di pysastrawi
        3. sentence segmentation ama tokenization apaan anjir wlwwkwkwkwkww
        """
        #Stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        #Stopword Remover
        factory2 = StopWordRemoverFactory()
        stopword = factory2.create_stop_word_remover()

        td_pairs = []
        
        directory = os.path.join(self.data_dir, block_dir_relative)
        for filename in os.listdir(directory):
            file = os.path.join(directory, filename)
            with open(file) as f:
                contents = f.read()
                output = stopword.remove(contents) #Remove stopwords
                output = stemmer.stem(output) #Stem
                res = output.split() #Tokenization
                for word in res:
                    td_pairs.append((self.term_id_map[word], self.doc_id_map[file]))
        return td_pairs

    def invert_write(self, td_pairs, index):
        """
        Melakukan inversion td_pairs (list of <termID, docID> pairs) dan
        menyimpan mereka ke index. Disini diterapkan konsep BSBI dimana 
        hanya di-mantain satu dictionary besar untuk keseluruhan block.
        Namun dalam teknik penyimpanannya digunakan srategi dari SPIMI
        yaitu penggunaan struktur data hashtable (dalam Python bisa
        berupa Dictionary)

        ASUMSI: td_pairs CUKUP di memori

        Parameters
        ----------
        td_pairs: List[Tuple[Int, Int]]
            List of termID-docID pairs
        index: InvertedIndexWriter
            Inverted index pada disk (file) yang terkait dengan suatu "block"
        """
        term_dict = {}
        for term_id, doc_id in td_pairs:
            if term_id not in term_dict:
                term_dict[term_id] = set()
            term_dict[term_id].add(doc_id)
        for term_id in sorted(term_dict.keys()):
            index.append(term_id, sorted(list(term_dict[term_id])))

    def merge(self, indices, merged_index):
        """
        Lakukan merging ke semua intermediate inverted indices menjadi
        sebuah single index.

        Ini adalah bagian yang melakukan EXTERNAL MERGE SORT

        Parameters
        ----------
        indices: List[InvertedIndexReader]
            A list of intermediate InvertedIndexReader objects, masing-masing
            merepresentasikan sebuah intermediate inveted index yang iterable
            di sebuah block.

        merged_index: InvertedIndexWriter
            Instance InvertedIndexWriter object yang merupakan hasil merging dari
            semua intermediate InvertedIndexWriter objects.
        """
        #Reference: https://notebooks.githubusercontent.com/view/ipynb?browser=chrome&color_mode=auto&commit=4912ce487c4562da6b73447366a5fd421df7ac07&device=unknown&enc_url=68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f5a68656e67787572752f43533237362d7061312d736b656c65746f6e2d323031392f343931326365343837633435363264613662373334343733363661356664343231646637616330372f5041312d736b656c65746f6e2e6970796e62&logged_in=false&nwo=Zhengxuru%2FCS276-pa1-skeleton-2019&path=PA1-skeleton.ipynb&platform=android&repository_id=299051363&repository_type=Repository&version=104
        term_before = ''
        postings_before = []
        for term_id, postings in heapq.merge(*indices, key=lambda x: x[0]):
            if term_id != term_before:
                if term_before != '':
                    merged_index.append(term_before, postings_before)
                term_before = term_id
                postings_before = postings
            else:
                pointer1, pointer2 = 0, 0
                new_postings = []
                while pointer1 < len(postings) and pointer2 < len(postings_before):
                    if postings[pointer1] < postings_before[pointer2]:
                        new_postings.append(postings[pointer1])
                        pointer1 += 1
                    elif postings[pointer1] > postings_before[pointer2]:
                        new_postings.append(postings_before[pointer2])
                        pointer2 += 1
                    else:
                        new_postings.append(postings[pointer1])
                        pointer1 += 1
                        pointer2 += 1
                if pointer1 < len(postings):
                    new_postings.extend(postings[pointer1:len(postings)])
                elif pointer2 < len(postings_before):
                    new_postings.extend(postings[pointer2:len(postings_before)])
                postings_before = new_postings
                del new_postings
        if term_before:
            merged_index.append(term_before, postings_before)  

    def retrieve(self, query):
        """
        Melakukan boolean retrieval untuk mengambil semua dokumen yang
        mengandung semua kata pada query. Jangan lupa lakukan pre-processing
        yang sama dengan yang dilakukan pada proses indexing!
        (Stemming dan Stopwords Removal)

        Parameters
        ----------
        query: str
            Query tokens yang dipisahkan oleh spasi

            contoh: Query "universitas indonesia depok" artinya adalah
                    boolean query "universitas AND indonesia AND depok"

        Result
        ------
        List[str]
            Daftar dokumen terurut yang mengandung sebuah query tokens.
            Harus mengembalikan EMPTY LIST [] jika tidak ada yang match.

        JANGAN LEMPAR ERROR/EXCEPTION untuk terms yang TIDAK ADA di collection.
        """
        if len(self.term_id_map) == 0 or len(self.doc_id_map) == 0:
            self.load()

        #Stemmer
        factory = StemmerFactory()
        stemmer = factory.create_stemmer()

        #Stopword Remover
        factory2 = StopWordRemoverFactory()
        stopword = factory2.create_stop_word_remover()

        output = stopword.remove(query)
        output = stemmer.stem(output)
        list_of_terms = output.split()
        list_of_docs = []
        list_of_postings_list = []
        list_of_docIDs = []
        
        #Jika query hanya 1 kata
        if len(list_of_terms) == 1:
            termID = self.term_id_map[list_of_terms[0]]
            with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
                try:
                    postings_list = index.get_postings_list(termID)
                except KeyError:
                    postings_list = []
                if len(postings_list) == 0:
                    list_of_docs = []
                else:
                    for docID in postings_list:
                        list_of_docs.append(self.doc_id_map[docID])
        #Jika query > 1 kata
        else:
            for term in list_of_terms:
                termID = self.term_id_map[term]
                with InvertedIndexReader(self.index_name, self.postings_encoding, directory = self.output_dir) as index:
                    try:
                        postings_list = index.get_postings_list(termID)
                    except KeyError:
                        postings_list = []    
                    list_of_postings_list.append(postings_list)
            
            #Boolean retrieval dari postings_list berukuran terkecil
            list1 = min(list_of_postings_list, key=len)
            list_of_postings_list.remove(min(list_of_postings_list, key=len))
            list2 = min(list_of_postings_list, key=len)
            list_of_postings_list.remove(min(list_of_postings_list, key=len))
            list_of_docIDs = sorted_intersect(list1, list2)

            while len(list_of_postings_list) > 0:
                list1 = min(list_of_postings_list, key=len)
                list_of_postings_list.remove(min(list_of_postings_list, key=len))
                list_of_docIDs = sorted_intersect(list_of_docIDs, sorted_intersect(list1, list2))

            for docID in list_of_docIDs:
                list_of_docs.append(self.doc_id_map[docID])

        return list_of_docs


    def index(self):
        """
        Base indexing code
        BAGIAN UTAMA untuk melakukan Indexing dengan skema BSBI (blocked-sort
        based indexing)

        Method ini scan terhadap semua data di collection, memanggil parse_block
        untuk parsing dokumen dan memanggil invert_write yang melakukan inversion
        di setiap block dan menyimpannya ke index yang baru.
        """
        # loop untuk setiap sub-directory di dalam folder collection (setiap block)
        for block_dir_relative in tqdm(sorted(next(os.walk(self.data_dir))[1])):
            td_pairs = self.parse_block(block_dir_relative)
            index_id = 'intermediate_index_'+block_dir_relative
            self.intermediate_indices.append(index_id)
            with InvertedIndexWriter(index_id, self.postings_encoding, directory = self.output_dir) as index:
                self.invert_write(td_pairs, index)
                td_pairs = None
    
        self.save()

        with InvertedIndexWriter(self.index_name, self.postings_encoding, directory = self.output_dir) as merged_index:
            with contextlib.ExitStack() as stack:
                indices = [stack.enter_context(InvertedIndexReader(index_id, self.postings_encoding, directory=self.output_dir))
                               for index_id in self.intermediate_indices]
                self.merge(indices, merged_index)


if __name__ == "__main__":

    BSBI_instance = BSBIIndex(data_dir = 'collection', \
                              postings_encoding = VBEPostings, \
                              output_dir = 'index')
    BSBI_instance.index() # memulai indexing!