import tkinter as tk
from tkinter import messagebox

# Create main window
root = tk.Tk()
root.title("IT352 Course Project")
root.geometry("850x600")
root.configure(bg="#eef5d7")  # Light background color

# Title Label
title_label = tk.Label(
    root, text="DEPARTMENT OF INFORMATION TECHNOLOGY\n"
               "NATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA, SURATHKAL-575025",
    font=("Arial", 14, "bold"), bg="#eef5d7", justify="center"
)
title_label.pack(pady=10)

# Course Project Label
project_label = tk.Label(
    root, text="Information Assurance and Security (IT352) Course Project\n"
               "Title “DEPHIDES: Deep Learning Based Phishing Detection System”",
    font=("Arial", 12), bg="#eef5d7", justify="center"
)
project_label.pack(pady=5)

# Carried out by Label
student_label = tk.Label(
    root, text="Carried out by\n"
               "Nidhi Kumari (221IT047)\n"
               "Sneha Singh (221IT063)\n"
               "During Academic Session January – April 2025",
    font=("Arial", 12, "bold"), bg="#eef5d7", justify="center"
)
student_label.pack(pady=10)

# Function for button actions
def enter_input():
    messagebox.showinfo("Input", "Enter Input Clicked!")

def display_output():
    messagebox.showinfo("Output", "Display Output Clicked!")

def store_output():
    messagebox.showinfo("Store", "Store Output Clicked!")

# Button Styling
button_style = {"font": ("Arial", 12, "bold"), "bg": "#4a90e2", "fg": "white", "width": 30, "height": 2}

# Buttons
btn1 = tk.Button(root, text="Press here to Enter Input", command=enter_input, **button_style)
btn1.pack(pady=10)

btn2 = tk.Button(root, text="Press here to display output on Screen", command=display_output, **button_style)
btn2.pack(pady=10)

btn3 = tk.Button(root, text="Press here to store the output", command=store_output, **button_style)
btn3.pack(pady=10)

# Run application
root.mainloop()
