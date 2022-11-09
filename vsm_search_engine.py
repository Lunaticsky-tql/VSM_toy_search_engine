"""
    Author: Lunaticsky
    Modified: 2022-11-8
"""
import datetime
import os
from queue import Queue
from threading import Thread
from tkinter.filedialog import askdirectory
import ttkbootstrap as ttk
from ttkbootstrap.constants import *
from ttkbootstrap import utility
from collections import defaultdict
from math import log

from ttkbootstrap.dialogs import Messagebox

# the two map are actually list, using array index as docID
id_title_map = []
id_author_map = []
word_bag = set()
inverted_index = defaultdict(list)
idf_dict = []
dataset_pos = ''
doc_number = 0
term_number = 0
doc_vectors = []
doc_vectors_len = []


class SearchEngine(ttk.Frame):
    queue = Queue()
    searching = False
    has_result = False

    def __init__(self, master):
        super().__init__(master, padding=15)
        self.content_textarea = None
        self.search_btn = None
        self.searching_thread = None
        self.result_view = None
        self.pack(fill=BOTH, expand=YES)

        # application variables
        self.title_var = ttk.StringVar(value='')
        self.author_var = ttk.StringVar(value='')
        self.option_var = ttk.StringVar(value='')

        # header and labelframe option container
        top_text = "Fill the form to begin your search"
        self.top_lf = ttk.Labelframe(self, text=top_text, padding=15)
        self.top_lf.pack(fill=X, expand=YES, anchor=N)

        self.create_title_row()
        self.create_author_row()
        self.create_content_row()
        self.create_option_row()
        self.create_results_view()

        self.progressbar = ttk.Progressbar(
            master=self,
            mode=INDETERMINATE,
            bootstyle=(STRIPED, SUCCESS)
        )
        self.progressbar.pack(fill=X, expand=YES)

    def create_title_row(self):
        """Enter article title"""
        title_row = ttk.Frame(self.top_lf)
        title_row.pack(fill=X, expand=YES)
        path_lbl = ttk.Label(title_row, text="Title", width=8)
        path_lbl.pack(side=LEFT, padx=(15, 0))
        title_ent = ttk.Entry(title_row, textvariable=self.title_var)
        title_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

    def create_author_row(self):
        """Enter article author"""
        author_row = ttk.Frame(self.top_lf)
        author_row.pack(fill=X, expand=YES, pady=15)
        author_lbl = ttk.Label(author_row, text="Author", width=8)
        author_lbl.pack(side=LEFT, padx=(15, 0))
        term_ent = ttk.Entry(author_row, textvariable=self.author_var)
        term_ent.pack(side=LEFT, fill=X, expand=YES, padx=5)

    def create_content_row(self):
        """Enter article content"""
        # create a textarea for the content
        content_row = ttk.Frame(self.top_lf)
        content_row.pack(fill=X, expand=YES)
        content_lbl = ttk.Label(content_row, text="Content", width=8)
        content_lbl.pack(side=LEFT, padx=(15, 0))
        self.content_textarea = ttk.Text(content_row, height=5)
        self.content_textarea.pack(fill=X, expand=YES, padx=5)

    def create_option_row(self):
        """Create the search options"""
        option_row = ttk.Frame(self.top_lf)
        option_row.pack(fill=X, expand=YES, pady=15)
        option_lbl = ttk.Label(option_row, text="Option", width=8)
        option_lbl.pack(side=LEFT, padx=15)
        precise_opt = ttk.Radiobutton(
            master=option_row,
            text="Precise",
            variable=self.option_var,
            value="precise"
        )
        precise_opt.pack(side=LEFT)
        fuzzy_opt = ttk.Radiobutton(
            master=option_row,
            text="Fuzzy",
            variable=self.option_var,
            value="fuzzy"
        )
        fuzzy_opt.pack(side=LEFT, padx=15)
        fuzzy_opt.invoke()
        self.search_btn = ttk.Button(
            master=option_row,
            text="Search",
            command=self.on_search,
            width=10
        )
        self.search_btn.pack(side=RIGHT, padx=15)

    def create_results_view(self):
        """Add result treeview to labelframe"""
        self.result_view = ttk.Treeview(
            master=self,
            bootstyle=INFO,
            columns=[0, 1],
            show=HEADINGS
        )
        self.result_view.pack(fill=BOTH, expand=YES, pady=10)

        # setup columns and use `scale_size` to adjust for resolution
        self.result_view.heading(0, text='Name')
        self.result_view.heading(1, text='Author')
        self.result_view.column(
            column=0,
            anchor=W,
            width=utility.scale_size(self, 120),
            stretch=True
        )
        self.result_view.column(
            column=1,
            anchor=E,
            width=utility.scale_size(self, 120),
            stretch=True
        )

    def show_warning(self):
        """Show warning message"""
        Messagebox.show_warning(
            title="Error",
            message="We can't find any results for your search",
            parent=self,
        )

    def show_error(self):
        """Show error message"""
        Messagebox.show_error(
            title="Error",
            message="Please enter at least one search term",
            parent=self,
        )

    def on_search(self):
        """Search for a term based on the search type"""
        search_title = self.title_var.get()
        search_author = self.author_var.get()
        search_content = self.content_textarea.get(1.0, END)
        search_option = self.option_var.get()

        if search_title == '' and search_author == '' and search_content == '':
            self.show_error()
            return

        # start search in another thread to prevent UI from locking
        self.searching_thread = Thread(
            target=SearchEngine.information_retrieval,
            args=(self, search_author, search_title, search_content, search_option),
            daemon=True
        ).start()
        SearchEngine.searching = True
        self.search_btn.configure(state=DISABLED)
        self.progressbar.start(10)
        self.listen_for_completion()

    def listen_for_completion(self):
        """Check if the upload task is complete"""
        listen_id = self.after(100, self.listen_for_completion)
        if not SearchEngine.searching:
            self.after_cancel(listen_id)
            self.progressbar.stop()
            self.search_btn.configure(state=NORMAL)
            if not SearchEngine.has_result:
                self.show_warning()

    def insert_row(self, name, author):
        """Insert a row into the treeview"""
        self.result_view.insert('', END, values=(name, author))

    def information_retrieval(self, author, title, content, option):
        """Search for articles based on the search type"""
        # clear the results
        for row in self.result_view.get_children():
            self.result_view.delete(row)

        # search for the term
        if option == 'precise':
            results = self.search_precise(author, title, content)
        else:
            results = self.search_fuzzy(author, title, content)

        # display the results
        if results:
            for result in results:
                # get the article name and author according to the id
                article = id_title_map[result]
                author = id_author_map[result]
                self.insert_row(article, author)
            SearchEngine.has_result = True
        else:
            SearchEngine.has_result = False
        SearchEngine.searching = False

    def search_precise(self, author, title, content):
        """Search for a precise term"""
        # for precise search, we need to make sure all terms are present
        title_based_results = []
        author_based_results = []
        precise_results = []
        if title:
            title_based_results = self.search_title(title, fuzzy=False)
        if author:
            author_based_results = self.search_author(author, fuzzy=False)
        # get the candidate doc_id
        if title and author:
            candidate_doc_id = list(set(title_based_results) & set(author_based_results))
        elif title and not author:
            candidate_doc_id = title_based_results
        elif author and not title:
            candidate_doc_id = author_based_results
        else:
            # only have content, all doc_id are candidates
            candidate_doc_id = list(range(doc_number))
        # search for content
        # parse the content for raw content
        content_input = content.replace('\n', ' ').replace('\t', ' ')
        if content_input.strip():
            # tokenize the content
            content_tokens = tokenize(list(content_input.split(' ')))
            candidate_doc_id = self.search_content(content_tokens, candidate_doc_id, fuzzy=False)
            # in precise search, we need to make sure all terms are present
            for doc_id in candidate_doc_id:
                if self.has_all_terms(doc_id, content_tokens):
                    precise_results.append(doc_id)
        else:
            precise_results = candidate_doc_id
        return precise_results

    @staticmethod
    def has_all_terms(doc_id, content_tokens):
        """Check if a document contains all terms"""
        doc_title = id_title_map[doc_id] + ".txt"
        # dataset_pos join dataset_title +.txt
        doc_path = os.path.join(dataset_pos, doc_title)
        with open(doc_path, 'r') as f:
            doc_content = f.read()
        for token in content_tokens:
            if token not in doc_content:
                return False
        return True

    def search_fuzzy(self, author, title, content):
        """Search for a fuzzy term"""
        # for fuzzy search, we want to return results as much as possible
        title_based_results = []
        author_based_results = []
        precise_results = []
        content_input = content.replace('\n', ' ').replace('\t', ' ')
        if content_input.strip():
            candidate_doc_id = list(range(doc_number))
            # tokenize the content
            content_tokens = tokenize(list(content_input.split(' ')))
            candidate_doc_id = self.search_content(content_tokens, candidate_doc_id, fuzzy=True)
        else:
            if title:
                title_based_results = self.search_title(title, fuzzy=True)
            if author:
                author_based_results = self.search_author(author, fuzzy=True)
            # get the candidate doc_id
            if title and author:
                # union of title and author candidates
                candidate_doc_id = list(set(title_based_results) | set(author_based_results))
            elif title and not author:
                candidate_doc_id = title_based_results
            else:
                # author and not title
                candidate_doc_id = author_based_results
        return candidate_doc_id

    def search_author(self, author, fuzzy=False):
        """Search for an author"""
        results = []
        # for docID in id_author_map:
        #     if fuzzy:
        #         if author in id_author_map[docID]:
        #             results.append(docID)
        #     else:
        #         if author == id_author_map[docID]:
        #             results.append(docID)
        if fuzzy:
            for docID in range(len(id_author_map)):
                if author in id_author_map[docID]:
                    results.append(docID)
        else:
            for docID in range(len(id_author_map)):
                if author == id_author_map[docID]:
                    results.append(docID)
        return results

    def search_title(self, author, fuzzy=False):
        """Search for a title"""
        results = []
        if fuzzy:
            for docID in range(len(id_title_map)):
                if author in id_title_map[docID]:
                    results.append(docID)
        else:
            for docID in range(len(id_title_map)):
                if author == id_title_map[docID]:
                    results.append(docID)
        return results

    def search_content(self, content_tokens, candidate_doc_id, fuzzy=False):
        """Search for a content"""
        # get the space vector of the query
        query_vector = [0] * term_number
        # use word_bag set as a list and order the terms
        word_bag_list = sorted(list(word_bag))
        for token in content_tokens:
            if token in word_bag_list:
                query_vector[word_bag_list.index(token)] += 1
        query_vector = [log(x + 1) for x in query_vector]
        for termID in range(term_number):
            if query_vector[termID] > 0:
                query_vector[termID] *= idf_dict[termID]
        # rank the candidate doc_id
        doc_score = []
        for docID in candidate_doc_id:
            score = self.cosine_similarity(query_vector, doc_vectors[docID], docID)
            doc_score.append((docID, score))
        doc_score.sort(key=lambda x: x[1], reverse=True)
        # print the score to the console to test the correctness
        print("{0: <40} {1: <40} {2: <40}".format("Title", "DocID", "Score"))
        for docID, score in doc_score:
            doc_name = id_title_map[docID]
            # print and set static width
            print("{0: <40} {1: <40} {2: <40}".format(doc_name, docID, score))
        result_rank = []
        for docID, score in doc_score:
            if score > 0:
                result_rank.append(docID)
        return result_rank

    @staticmethod
    def cosine_similarity(vector1, vector2, docID):
        """Calculate the cosine similarity between two vectors"""
        numerator = sum([vector1[i] * vector2[i] for i in range(len(vector1))])
        # do not need to divide query vector by its length since we are only comparing
        denominator = doc_vectors_len[docID]
        if denominator == 0:
            return 0
        else:
            return numerator / denominator


def middle_window(root, width=480, height=480):
    """Center window on screen"""
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    root.geometry(f'{width}x{height}+{int(x)}+{int(y)}')


def doc_pre_processing():
    """Read files, create idmap, and tokenize words"""
    working_dir = os.getcwd()
    global dataset_pos, word_bag, doc_number, term_number
    dataset_pos = os.path.join(working_dir, 'dataset')
    print(f"Dataset directory: {dataset_pos}")
    docID = 0
    for file in os.listdir(dataset_pos):
        with open(os.path.join(dataset_pos, file), 'r') as f:
            text = f.read()
            # save the file name to the id_title map
            file_name = file.split('.')[0]
            id_title_map.insert(docID, file_name)
            # get the author
            author = text.splitlines()[0].split(':')[1].strip()
            id_author_map.insert(docID, author)
            content = text.splitlines()[1:]
            # tokenize the content
            tokens = tokenize(content)
            unique_tokens = set(tokens)
            word_bag = word_bag.union(unique_tokens)
            # create the inverted index
            for term in unique_tokens:
                if term in inverted_index:
                    # insert docID and term frequency to the posting list
                    inverted_index[term].append((docID, log(1 + tokens.count(term))))
                else:
                    inverted_index[term] = [(docID, log(1 + tokens.count(term)))]
            docID += 1
    doc_number = docID
    term_number = len(word_bag)
    print(f"Total number of documents: {doc_number}")
    print(f"Total number of words: {term_number}")


def tokenize(text):
    """Tokenize text"""
    stop_chars = ".,;:?!-'"  # remove stop characters
    stop_words = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its',
                  'of', 'on', 'that', 'the', 'to', 'was', 'were', 'will', 'with']
    tokens = []
    for line in text:
        for word in line.split():
            word = word.strip(stop_chars).lower()
            if word not in stop_words and word.isalpha():
                tokens.append(word)
    return tokens


def calculate_tf_idf():
    """Calculate idf for each term in each document"""
    global doc_vectors, idf_dict, doc_vectors_len
    # initialize the doc_vectors with dxt matrix
    for i in range(doc_number):
        doc_vectors.append([0] * term_number)
    termID = 0
    for term in sorted(inverted_index):
        # calculate the idf
        idf = log(doc_number / len(inverted_index[term]))
        idf_dict.append(idf)
        for doc_tf_tuple in inverted_index[term]:
            doc_id = doc_tf_tuple[0]
            tf = doc_tf_tuple[1]
            # calculate the tf-idf
            tf_idf = tf * idf
            # save the tf-idf to the doc_vectors
            doc_vectors[doc_id][termID] = tf_idf
        termID += 1
    # calculate the length of each document vector offline
    for docID in range(doc_number):
        doc_vectors_len.append(sum([x ** 2 for x in doc_vectors[docID]]))
        # do not use sqrt to save time
        # doc_vectors[docID] = sqrt(sum(doc_vectors[docID]))
    print("TF-IDF calculation completed")


def EngineInit():
    """Initialize search engine"""
    doc_pre_processing()
    calculate_tf_idf()


if __name__ == '__main__':
    app = ttk.Window("My Search Engine", "journal")
    app.eval('tk::PlaceWindow . center')
    middle_window(app)
    EngineInit()
    SearchEngine(app)
    app.mainloop()
