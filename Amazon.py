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
from surprise import Reader, Dataset
from surprise import KNNBasic

# Fungsi untuk membuka file CSV
def open_file():
    global df  # Agar DataFrame dapat diakses oleh fungsi lain
    # Memilih file CSV
    file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
    if file_path:
        try:
            # Membaca file CSV dengan pandas
            df = pd.read_csv(file_path)
            label_file.config(text=f"Loaded: {file_path}")
            display_data(df)  # Menampilkan data di GUI
        except Exception as e:
            label_file.config(text="Error loading file")
            print(f"Error: {e}")

# Fungsi untuk menampilkan data di Treeview
def display_data(df):
    # Membersihkan data sebelumnya dari Treeview
    for widget in frame_tree.winfo_children():
        widget.destroy()

    # Membuat Treeview
    tree = ttk.Treeview(frame_tree, show="headings")
    
    # Menambahkan kolom ke Treeview
    tree["columns"] = list(df.columns)

    # Menambahkan header kolom
    for col in df.columns:
        tree.heading(col, text=col)  # Header kolom
        tree.column(col, anchor="center", width=120)  # Atur lebar kolom
    
    # Menambahkan data ke Treeview
    for _, row in df.iterrows():
        tree.insert("", "end", values=list(row))

    # Menambahkan scrollbar vertikal
    v_scrollbar = ttk.Scrollbar(frame_tree, orient="vertical", command=tree.yview)
    v_scrollbar.pack(side="right", fill="y")
    tree.configure(yscrollcommand=v_scrollbar.set)

    # Menambahkan scrollbar horizontal
    h_scrollbar = ttk.Scrollbar(frame_tree, orient="horizontal", command=tree.xview)
    h_scrollbar.pack(side="bottom", fill="x")
    tree.configure(xscrollcommand=h_scrollbar.set)

    # Menambahkan Treeview ke dalam canvas untuk scrolling
    tree.pack(fill="both", expand=True)

# Fungsi untuk menampilkan grafik kolom Primary Categories
def Primary_Categories():
    try:
        if 'primaryCategories' not in df.columns:
            label_file.config(text="Error: Kolom 'primaryCategories' tidak ditemukan")
            return

        primary_cat = df['primaryCategories'].value_counts()
        
        # Membuat grafik dengan Matplotlib
        dims = (10, 8)
        fig, ax = plt.subplots(figsize=dims)
        sns.countplot(x='primaryCategories', data=df, order=primary_cat.index, ax=ax)
        plt.xticks(rotation=45)
        plt.title('Distribusi Primary Categories')
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error saat membuat grafik: {e}")

# Fungsi untuk merename kolom dan menampilkan grafik kolom reviewsRating
def Reviews_Rating():
    try:
        # Ubah nama kolom untuk mempermudah
        df.rename(columns={"reviews.rating": "reviewsRating"}, inplace=True)
        
        # Hitung nilai yang ada di kolom reviewsRating
        rev_rating = df['reviewsRating'].value_counts()
        print("Value counts for reviewsRating:")
        print(rev_rating)

        # Visualisasikan hasil perhitungan nilai
        dims = (10, 8)
        fig, ax = plt.subplots(figsize=dims)
        sns.countplot(x="reviewsRating", data=df)
        plt.title("Distribusi Rating Review")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        label_file.config(text="Error: Kolom 'reviews.rating' tidak ditemukan atau ada masalah lain")
        print(f"Error: {e}")

# Fungsi untuk merename kolom menjadi reviewsRecommend 
def Reviews_Recommend():
    try:
        # Ubah nama kolom untuk mempermudah
        df.rename(columns={"reviews.doRecommend": "reviewsRecommend"}, inplace=True)
        
        # Hitung nilai yang ada di kolom reviewsRecommend
        rev_rec = df['reviewsRecommend'].value_counts()
        print("Value counts for reviewsRecommend:")
        print(rev_rec)

        # Visualisasikan hasil perhitungan nilai
        dims = (10, 8)
        fig, ax = plt.subplots(figsize=dims)
        sns.countplot(x="reviewsRecommend", data=df)
        plt.title("Distribusi Review Recommend")
        plt.tight_layout()
        plt.show()
        
    except Exception as e:
        label_file.config(text="Error: Kolom 'reviews.doRecommend' tidak ditemukan atau ada masalah lain")
        print(f"Error: {e}")

# Fungsi untuk menampilkan grafik primaryCategories dengan reviewsRecommend
def primaryCategories_reviewsRecommend():
    try:
        if 'primaryCategories' not in df.columns or 'reviewsRecommend' not in df.columns:
            label_file.config(text="Error: Kolom 'primaryCategories' atau 'reviewsRecommend' tidak ditemukan")
            return
        
        # Membuat grafik dengan Matplotlib menggunakan hue untuk reviewsRecommend
        dims = (10, 8)
        fig, ax = plt.subplots(figsize=dims)
        sns.countplot(x="primaryCategories", hue="reviewsRecommend", data=df, ax=ax)
        plt.xticks(rotation=45)
        plt.title('Distribusi Primary Categories dengan Reviews Recommend')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error saat membuat grafik dengan hue: {e}")

# Fungsi data preprocessing 
def Data_Preprocessing():
    try:
        # Mengecek apakah ada nilai yang null dalam dataset
        null_counts = df.isnull().sum()

        # Menyiapkan dataset untuk train-test split dengan framework Surprise
        df_subset = df[['reviews.username', 'id', 'reviewsRating']]

        # Menyiapkan hasil untuk ditampilkan di GUI
        result_text = "Jumlah Nilai Null pada Setiap Kolom:\n"
        result_text += null_counts.to_string() + "\n\n"
        result_text += "Dataset yang akan digunakan untuk train-test split:\n"
        result_text += df_subset.head().to_string()

        # Menampilkan hasil preprocessing di dalam Text widget
        text_result.config(state=tk.NORMAL)  # Membuka Text widget untuk diubah
        text_result.delete(1.0, tk.END)  # Menghapus hasil sebelumnya
        text_result.insert(tk.END, result_text)  # Menampilkan hasil baru
        text_result.config(state=tk.DISABLED)  # Membuat Text widget menjadi read-only

    except Exception as e:
        label_file.config(text="Error dalam preprocessing data.")
        print(f"Error: {e}")

# Fungsi split data
def Split_Data():
    try:
        # Membuat objek Reader untuk load dataset ke Surprise
        reader = Reader()
        data = Dataset.load_from_df(df[['reviews.username', 'id', 'reviewsRating']], reader)

        # Melakukan Train-Test Split
        trainset, testset = train_test_split(data, test_size=0.20, random_state=50)

        # Menampilkan hasil Train-Test Split
        result_text = "Trainset:\n"
        result_text += f"Number of rows in trainset: {len(trainset)}\n\n"
        result_text += "Testset:\n"
        result_text += f"Number of rows in testset: {len(testset)}\n"

        # Menampilkan hasil split di dalam Text widget
        text_result.config(state=tk.NORMAL)  # Membuka Text widget untuk diubah
        text_result.delete(1.0, tk.END)  # Menghapus hasil sebelumnya
        text_result.insert(tk.END, result_text)  # Menampilkan hasil baru
        text_result.config(state=tk.DISABLED)  # Membuat Text widget menjadi read-only

    except Exception as e:
        label_file.config(text="Error dalam split data.")
        print(f"Error: {e}")


# Membuat GUI utama
root = tk.Tk()
root.title("CSV Viewer")
root.geometry("800x600")

# Label untuk status file
label_file = tk.Label(root, text="No file selected", anchor="center")
label_file.pack(pady=10)

# Frame untuk tombol (horizontal layout)
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

# Tombol untuk membuka file
btn_open = tk.Button(button_frame, text="Buka File", command=open_file)
btn_open.pack(side="left", padx=5)

# Tombol untuk membuka grafik Primary Categories
btn_graph = tk.Button(button_frame, text="Grafik Primary Categories", command=Primary_Categories)
btn_graph.pack(side="left", padx=5)

# Tombol untuk rename kolom dan menampilkan grafik reviewsRating
btn_reviews = tk.Button(button_frame, text="Grafik Rating Review", command=Reviews_Rating)
btn_reviews.pack(side="left", padx=5)

# Tombol untuk rename kolom reviewsRecommend dan menampilkan grafik
btn_recommend = tk.Button(button_frame, text="Grafik Review Recommend", command=Reviews_Recommend)
btn_recommend.pack(side="left", padx=5)

# Tombol untuk menampilkan grafik primaryCategories dengan hue reviewsRecommend
btn_graph_with_hue = tk.Button(button_frame, text="Grafik Kombinasi", command=primaryCategories_reviewsRecommend)
btn_graph_with_hue.pack(side="left", padx=5)

# Tombol untuk data preprocessing
btn_preprocessing = tk.Button(button_frame, text="Data Preprocessing", command=Data_Preprocessing)
btn_preprocessing.pack(side="left", padx=5)

# Frame untuk hasil preprocessing
frame_preprocess = tk.Frame(root)
frame_preprocess.pack(fill="both", expand=False, pady=20, side="bottom")

# Text widget untuk menampilkan hasil preprocessing
text_result = tk.Text(frame_preprocess, height=10, width=80)
text_result.pack(pady=10)
text_result.config(state=tk.DISABLED)  # Membuat Text widget menjadi read-only

# Tombol untuk split data
btn_split = tk.Button(button_frame, text="Split Data", command=Split_Data)
btn_split.pack(side="left", padx=5)


# # Frame untuk menampilkan hasil preprocessing
# frame_result = tk.Frame(root)
# frame_result.pack(fill="both", expand=True, pady=20)

# # Text widget untuk menampilkan hasil preprocessing
# text_result = tk.Text(frame_result, height=10, width=90)
# text_result.pack()

# Frame untuk Treeview (scrollable)
frame_tree = tk.Frame(root)
frame_tree.pack(fill="both", expand=True)

# Menjalankan GUI
root.mainloop()