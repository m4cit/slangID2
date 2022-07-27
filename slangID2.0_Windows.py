import tkinter as tk
from classifiers.ignore_predict import train_svm, predictor_svm
from classifiers.ignore_predict import train_log, predictor_log
from classifiers.ignore_predict import train_dt, predictor_dec
from classifiers.ignore_predict import train_nbg, predictor_naive_g
from classifiers.ignore_predict import train_nbm, predictor_naive_m
from classifiers.ignore_predict import predictor_all, train_all
from classifiers.log_regression import dm_log_reg, dm_train_log
from classifiers.linear_svm import dm_lin_svm, dm_train_svm
from classifiers.decision_tree import dm_dec_tree, dm_train_dt
from classifiers.gaussian_nb import dm_naive_g, dm_train_nbg
from classifiers.multinomial_nb import dm_naive_m, dm_train_nbm
import os
import platform

def dm_train_all():
    dm_train_svm()
    dm_train_dt()
    dm_train_nbg()
    dm_train_nbm()
    dm_train_log()

def dm_predict_all():
    dm_lin_svm()
    dm_dec_tree()
    dm_naive_g()
    dm_naive_m()
    dm_log_reg()

def clear():
    if platform.system() == "Windows":
        os.system("CLS")
    elif platform.system() == "Linux":
        os.system("clear")

root = tk.Tk()

canvas = tk.Canvas(root, height=450, width=800)
canvas.pack()

frame = tk.Frame(root, bg="white")
frame.place(relwidth=1, relheight=1)

phrase_entry = tk.Text(root, relief="sunken", bd=5, font=("Calibri", 16))
phrase_entry.place(height=100, width=760, x=20, y=50)

logo_image = tk.PhotoImage(file="misc/slangID_Logo_bg.png")

logo_bg = tk.Label(frame, image=logo_image)
logo_bg.place(anchor="n", x=397, y=-3)

# Labels
#################################################################################################################################
svm_label = tk.Label(frame, text="Linear SVM", bg="white", font=("Calibri", 14), relief="ridge")
svm_label.place(width=200, height=50, x=20, y=172)

dt_label = tk.Label(frame, text="Decision Tree", bg="white", font=("Calibri", 14), relief="ridge")
dt_label.place(width=200, height=50, x=20, y=227)

nbg_label = tk.Label(frame, text="Naive Bayes (Gaussian)", bg="white", font=("Calibri", 13),relief="ridge")
nbg_label.place(width=200, height=50, x=20, y=282)

nbm_label = tk.Label(frame, text="Naive Bayes (Multinomial)", bg="white", font=("Calibri", 13),relief="ridge")
nbm_label.place(width=200, height=50, x=442, y=172)

log_label = tk.Label(frame, text="Logistic Regression", bg="white", font=("Calibri", 14), relief="ridge")
log_label.place(width=200, height=50, x=442, y=227)

all_label = tk.Label(frame, text="All models", bg="#fe9d3f", font=("Calibri", 14), relief="ridge")
all_label.place(width=200, height=50, x=442, y=282)

entry_label = tk.Label(frame, bg="#a0a0a0", fg="white", text="Give me a sentence", font=("Calibri", 14, "bold"))
entry_label.place(x=20, y=21)
#################################################################################################################################

# Buttons
################################################################################################################################
use_svm_button = tk.Button(root, command=lambda: predictor_svm([phrase_entry.get("1.0",tk.END)]), text="use", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
use_svm_button.place(width=70, height=50, x=219, y=172)
train_svm_button = tk.Button(root, command=lambda: train_svm(), text="train", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
train_svm_button.place(width=70, height=50, x=289, y=172)

use_dt_button = tk.Button(root, command=lambda: predictor_dec([phrase_entry.get("1.0",tk.END)]), text="use", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
use_dt_button.place(width=70, height=50, x=219, y=227)
train_dt_button = tk.Button(root, command=lambda: train_dt(), text="train", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
train_dt_button.place(width=70, height=50, x=289, y=227)

use_nbg_button = tk.Button(root, command=lambda: predictor_naive_g([phrase_entry.get("1.0",tk.END)]), text="use", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
use_nbg_button.place(width=70, height=50, x=219, y=282)
train_nbg_button = tk.Button(root, command=lambda: train_nbg(), text="train", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
train_nbg_button.place(width=70, height=50, x=289, y=282)

use_nbm_button = tk.Button(root, command=lambda: predictor_naive_m([phrase_entry.get("1.0",tk.END)]), text="use", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
use_nbm_button.place(width=70, height=50, x=641, y=172)
train_nbm_button = tk.Button(root, command=lambda: train_nbm(), text="train", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
train_nbm_button.place(width=70, height=50, x=711, y=172)

use_log_button = tk.Button(root, command=lambda: predictor_log([phrase_entry.get("1.0",tk.END)]), text="use", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
use_log_button.place(width=70, height=50, x=641, y=227)
train_log_button = tk.Button(root, command=lambda: train_log(), text="train", bg="white", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
train_log_button.place(width=70, height=50, x=711, y=227)

use_all_button = tk.Button(root, command=lambda: predictor_all([phrase_entry.get("1.0",tk.END)]), text="use", bg="#fe9d3f", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
use_all_button.place(width=70, height=50, x=641, y=282)
train_all_button = tk.Button(root, command=lambda: train_all(), text="train", bg="#fe9d3f", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
train_all_button.place(width=70, height=50, x=711, y=282)

demo_button = tk.Button(root, command=lambda: dm_predict_all(), text="DEMO", fg="white", bg="black", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
demo_button.place(anchor="n", width=100, height=60, x=395, y=370)

clear_button = tk.Button(root, command=lambda: clear(), text="CLEAR OUTPUT", fg="black", bg="#bf0606", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
clear_button.place(anchor="nw", width=160, height=60, x=450, y=370)

demo_train_button = tk.Button(root, command=lambda: dm_train_all(), text="DEMO train all", fg="white", bg="black", font=("Calibri", 14,"bold"), padx=5, pady=5, relief="raised")
demo_train_button.place(anchor="ne", width=180, height=60, x=340, y=370)
################################################################################################################################

root.mainloop()
