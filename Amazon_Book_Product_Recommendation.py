import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import surprise
from surprise import SVD
from surprise import accuracy
from surprise.model_selection import train_test_split
from surprise import Reader, Dataset, SVD
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image, ImageTk  # Untuk menampilkan gambar buku

# Variabel global untuk menyimpan data yang dimuat
data_files = {}
complete_df = None  # Variabel global untuk menyimpan dataset gabungan
pt = None
similarity_score = None
books = pd.read_csv('Books.csv')
ratings = pd.read_csv('Ratings.csv')
users = pd.read_csv('Users.csv')

# Fungsi untuk membuka file dan ditampilkan di tab baru
def open_file():
    # Memilih file CSV
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            # Membaca file CSV dengan pandas
            df = pd.read_csv(file_path)
            file_name = file_path.split("/")[-1]
            data_files[file_name] = df  # Menyimpan DataFrame ke dictionary
            add_new_tab(df, file_name)  # Membuat tab baru untuk dataset
        except Exception as e:
            label_file.config(text="Error loading file")
            print(f"Error: {e}")

# Fungsi untuk menambahkan tab baru
def add_new_tab(df, tab_name):
    # Membuat frame untuk tab baru
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=tab_name)  # Nama tab diambil dari nama file

    # Membuat Treeview di tab baru
    tree = ttk.Treeview(frame, show="headings")
    tree["columns"] = list(df.columns)

    # Menambahkan header kolom
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=120)

    # Menambahkan data ke Treeview
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Menambahkan scrollbar vertikal
    v_scrollbar = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
    v_scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=v_scrollbar.set)

    # Menambahkan scrollbar horizontal
    h_scrollbar = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
    h_scrollbar.pack(side="bottom", fill="x")
    tree.configure(xscrollcommand=h_scrollbar.set)

    # Menempatkan Treeview ke frame
    tree.pack(fill="both", expand=True)

# Fungsi untuk analisis data dan ditampilkan di tab baru
def analyze_data():
    if not data_files:
        label_file.config(text="Tidak ada dataset")
        return

    # Hasil analisis untuk semua dataset
    analysis_results = []
    for file_name, df in data_files.items():
        analysis = {
            "Dataset": file_name,
            "Shape": df.shape,
            "Null Values": df.isnull().sum().sum(),
            "Duplicated Rows": df.duplicated().sum(),
        }
        analysis_results.append(analysis)

    # Membuat tab baru untuk hasil analisis data
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Hasil Analisis Data")

    # Membuat Treeview untuk menampilkan hasil analisis
    tree = ttk.Treeview(frame, show="headings")
    tree["columns"] = ["Dataset", "Shape", "Null Values", "Duplicated Rows"]

    # Menambahkan header kolom
    for col in tree["columns"]:
        tree.heading(col, text=col)
        tree.column(col, anchor="center", width=150)

    # Menambahkan hasil analisis ke Treeview
    for result in analysis_results:
        tree.insert("", "end", values=list(result.values()))

    # Menambahkan scrollbar
    tree.pack(fill="both", expand=True)

# Fungsi untuk menggabungkan dataset dan menyimpan hasilnya
def merge_datasets():
    global complete_df
    required_datasets = ["Books.csv", "Ratings.csv", "Users.csv"]
    for dataset in required_datasets:
        if dataset not in data_files:
            label_file.config(text=f"{dataset} belum dimuat")
            return

    try:
        # Memuat dataset dari variabel global
        books = data_files["Books.csv"]
        ratings = data_files["Ratings.csv"]
        users = data_files["Users.csv"]

        # Menggabungkan datasets
        ratings_with_book_titles = ratings.merge(books, on="ISBN")
        ratings_with_book_titles.drop(
            columns=["ISBN", "Image-URL-S", "Image-URL-M"], inplace=True
        )
        complete_df = ratings_with_book_titles.merge(
            users.drop("Age", axis=1), on="User-ID"
        )

        # Membuat tab baru untuk menampilkan hasil
        add_new_tab(complete_df, "Merged Data")

    except Exception as e:
        label_file.config(text="Error saat menggabungkan dataset")
        print(f"Error: {e}")

# Fungsi untuk collaborative filtering
def collaborative_filtering():
    global complete_df, pt, similarity_score
    if complete_df is None:
        label_file.config(text="Dataset gabungan belum tersedia.")
        return

    try:
        # Filter pengguna dengan lebih dari 200 rating
        min_ratings_threshold = 200
        num_ratings_per_user = complete_df.groupby('User-ID')['Book-Rating'].count()
        knowledgeable_user_ids = num_ratings_per_user[num_ratings_per_user > min_ratings_threshold].index
        knowledgeable_user_ratings = complete_df[complete_df['User-ID'].isin(knowledgeable_user_ids)]

        # Filter buku dengan lebih dari 50 rating
        min_ratings_count_threshold = 50
        rating_counts = knowledgeable_user_ratings.groupby('Book-Title').count()['Book-Rating']
        popular_books = rating_counts[rating_counts >= min_ratings_count_threshold].index
        final_ratings = knowledgeable_user_ratings[knowledgeable_user_ratings['Book-Title'].isin(popular_books)]

        # Membuat tabel pivot
        pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
        pt.fillna(0, inplace=True)

        # Menampilkan hasil di tab baru
        add_new_tab(pt, "Collaborative Filtering")
        print("Collaborative filtering executed successfully.")  # Debug print

        # Menghitung cosine similarity setelah pivot table
        similarity_score = cosine_similarity(pt)
        print("Cosine similarity calculated.")  # Debug print

    except Exception as e:
        label_file.config(text="Error saat proses collaborative filtering")
        print(f"Error: {e}")

# Fungsi untuk memberikan rekomendasi
def recommend(book_name):
    global pt, similarity_score, books
    if pt is None or similarity_score is None:
        label_file.config(text="Pivot table atau similarity score belum dihitung")
        return None

    if books is None:
        label_file.config(text="Dataset Books belum dimuat.")
        return None  # Menghentikan eksekusi jika books belum dimuat

    try:
        # Mencari buku berdasarkan judul atau pengarang yang cocok
        matched_books = books[books['Book-Title'].str.contains(book_name, case=False, na=False) | 
                              books['Book-Author'].str.contains(book_name, case=False, na=False)]
        
        if matched_books.empty:
            label_file.config(text="Buku tidak ditemukan dalam dataset")
            return None
        
        # Ambil salah satu buku yang ditemukan untuk perhitungan similarity
        book_title = matched_books.iloc[0]['Book-Title']
        
        # Mencari indeks buku yang sesuai dengan nama buku
        index = np.where(pt.index == book_title)[0][0]
        
        # Menghitung kesamaan dan mencari 5 buku paling mirip
        similar_books = sorted(
            list(enumerate(similarity_score[index])), key=lambda x: x[1], reverse=True
        )[1:6]  # Menghindari buku itu sendiri dengan [1:6]

        data = []
        for i in similar_books:
            item = []
            temp_df = books[books["Book-Title"] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values))
            item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values))
            item.extend(list(temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values))
            # item.extend(list(temp_df.drop_duplicates("Book-Title")["Year-Of-Publication"].values))
            data.append(item)
        
        return data
    except IndexError:
        label_file.config(text="Buku tidak ditemukan dalam dataset")
        return None

# Fungsi untuk melakukan pencarian dan menampilkan hasil
def search_recommendation():
    search_query = search_bar.get()
    if not search_query:
        label_file.config(text="Harap masukkan nama buku untuk mencari rekomendasi")
        return

    recommendations = recommend(search_query)
    if recommendations:
        display_recommendations(recommendations, search_query)
    else:
        label_file.config(text="Tidak ada rekomendasi yang ditemukan")

# Fungsi untuk menampilkan rekomendasi di tab baru
def display_recommendations(recommendations, book_name):
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=f"Rekomendasi: {book_name}")

    for item in recommendations:
        title, author, image_url = item

        # Label untuk judul buku
        lbl_title = tk.Label(frame, text=f"Judul: {title}", font=("Arial", 12, "bold"))
        lbl_title.pack(anchor="w", pady=5)

        # Label untuk penulis
        lbl_author = tk.Label(frame, text=f"Penulis: {author}", font=("Arial", 10))
        lbl_author.pack(anchor="w", pady=5)

        # Label untuk URL gambar buku
        lbl_url = tk.Label(frame, text=f"Image URL: {image_url}", font=("Arial", 10), fg="blue", cursor="hand2")
        lbl_url.pack(anchor="w", pady=5)
        lbl_url.bind("<Button-1>", lambda e, url=image_url: open_url(url))

# Fungsi untuk membuka URL di browser
def open_url(url):
    import webbrowser
    webbrowser.open_new(url)

# Fungsi untuk melatih model 
def train_model():
    global model, complete_df
    if complete_df is None or complete_df.empty:
        label_file.config(text="Dataset belum digabung")
        return

    try:
        # Mengatur skala rating
        reader = Reader(rating_scale=(0, 10))

        # Memuat dataset ke format Surprise
        data = Dataset.load_from_df(complete_df[['User-ID', 'Book-Title', 'Book-Rating']], reader)

        # Membagi dataset menjadi training dan testing
        train_set, test_set = train_test_split(data, test_size=0.20, random_state=42)

        # Inisialisasi dan pelatihan model SVD
        model = SVD()

        # Train data
        model.fit(train_set)

        # Membuat prediksi pada data testing
        predictions = model.test(test_set)

        # Menghitung nilai RMSE
        rmse_value = accuracy.rmse(predictions)

        # Menampilkan hasil RMSE di tab baru
        show_rmse_result(rmse_value)

        # Menampilkan pesan jika model berhasil dilatih
        label_file.config(text="Berhasil Menghitung RMSE")
    except Exception as e:
        label_file.config(text=f"Error melatih model: {str(e)}")

# Fungsi menampilkan hasil RMSE
def show_rmse_result(rmse_value):
    # Membuat tab baru untuk hasil RMSE
    frame = ttk.Frame(notebook)
    notebook.add(frame, text="Hasil Evaluasi RMSE")

    # Menampilkan nilai RMSE
    result_label = tk.Label(frame, text=f"RMSE: {rmse_value:.4f}", font=("Arial", 14))
    result_label.pack(pady=20)

# Fungsi menampilkan hasil rekomendasi user
def recommend_books(user_id, n=10):
    global model, complete_df

    if complete_df is None or model is None:
        label_file.config(text="Dataset atau model belum siap")
        return

    try:
        # List semua buku unik
        all_books = complete_df['Book-Title'].unique()

        # Buku yang sudah diberi rating oleh user
        rated_books = complete_df[complete_df['User-ID'] == user_id]['Book-Title'].values
        books_to_predict = [book for book in all_books if book not in rated_books]

        # Prediksi rating untuk buku yang belum dirating
        predictions = []
        for book in books_to_predict:
            pred = model.predict(user_id, book)
            predictions.append((book, pred.est))

        # Urutkan berdasarkan rating prediksi
        predictions.sort(key=lambda x: x[1], reverse=True)

        # Ambil N teratas
        top_n = predictions[:n]

        # Tampilkan hasil di tab baru
        display_recommendation_results(user_id, top_n)

    except Exception as e:
        label_file.config(text=f"Error saat rekomendasi: {str(e)}")

# Fungsi untuk menampilkan hasil rekomendasi di tab baru
def display_recommendation_results(user_id, recommendations):
    frame = ttk.Frame(notebook)
    notebook.add(frame, text=f"Rekomendasi untuk User {user_id}")

    # Tampilkan judul di tab
    title_label = tk.Label(frame, text=f"Top Rekomendasi untuk User {user_id}", font=("Arial", 14, "bold"))
    title_label.pack(pady=10)

    # Tampilkan daftar buku yang direkomendasikan
    for i, (title, estimated_rating) in enumerate(recommendations, start=1):
        book_label = tk.Label(frame, text=f"{i}. {title} (Est. Rating: {estimated_rating:.2f})", font=("Arial", 12))
        book_label.pack(anchor="w", padx=20, pady=5)

# Fungsi untuk mengambil user_id dari input dan memberikan rekomendasi
def get_user_recommendations():
    user_id = user_id_entry.get()  # Ambil user_id dari input
    if not user_id.isdigit():
        label_file.config(text="Masukkan User ID yang valid (hanya angka)")
        return

    recommend_books(int(user_id))  # Panggil fungsi rekomendasi

    
# Membuat GUI utama
root = tk.Tk()
root.title("AMAZON BOOK PRODUCT RECOMMENDATION")
root.geometry("800x600")

# Notebook untuk membuat tab
notebook = ttk.Notebook(root)
notebook.pack(fill="both", expand=True)

# Label untuk status file
label_file = tk.Label(root, text="No file selected", anchor="center")
label_file.pack(pady=10)

# Frame untuk tombol
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Tombol untuk membuka file
btn_open_file = tk.Button(button_frame, text="Buka File CSV", command=open_file)
btn_open_file.pack(side="left", padx=5)

# Tombol untuk menganalisis data
btn_analyze = tk.Button(button_frame, text="Analisis Data", command=analyze_data)
btn_analyze.pack(side="left", padx=5)

# Tombol untuk menggabungkan dataset
btn_merge = tk.Button(button_frame, text="Gabungkan Dataset", command=merge_datasets)
btn_merge.pack(side="left", padx=5)

# Tombol untuk collaborative filtering
btn_collab_filtering = tk.Button(button_frame, text="Collaborative Filtering", command=collaborative_filtering)
btn_collab_filtering.pack(side="left", padx=5)

# Search bar untuk mencari buku
search_bar = tk.Entry(button_frame, width=40)
search_bar.pack(side="left", padx=5)
btn_search = tk.Button(button_frame, text="Cari Rekomendasi", command=search_recommendation)
btn_search.pack(side="left", padx=5)

# Tombol untuk melatih model dan menampilkan hasil RMSE
btn_train_model = tk.Button(button_frame, text="Evaluasi RMSE", command=train_model)
btn_train_model.pack(side="left", padx=5)

# Tombol untuk rekomendasi dari id users
btn_user_rekomendasi = tk.Button(button_frame, text="Hasil Rekomendasi Untuk User", command=get_user_recommendations)
btn_user_rekomendasi.pack(side="left", padx=5)

# Input untuk User ID
user_id_entry = tk.Entry(button_frame, width=10)
user_id_entry.pack(side="left", padx=5)

# Menjalankan GUI
root.mainloop()