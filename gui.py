import tkinter as tk
from tkinter import font

def black_move():
    run_my_code("black")

def white_move():
    run_my_code("white")

def run_my_code(param: str):
    print(f"Running with param: {param}")
    # כאן תכניס את הקוד שלך

def on_enter_black(e):
    black_btn.config(bg="#333333")

def on_leave_black(e):
    black_btn.config(bg="#111111")

def on_enter_white(e):
    white_btn.config(bg="#dddddd")

def on_leave_white(e):
    white_btn.config(bg="#ffffff")

# חלון ראשי
window = tk.Tk()
window.title("Chess Moves")
window.geometry("400x300")
window.configure(bg="#1a1a2e")
window.resizable(False, False)

# כותרת
title_label = tk.Label(
    window,
    text="♟ Choose Your Move",
    bg="#1a1a2e",
    fg="#e0e0e0",
    font=("Georgia", 20, "bold")
)
title_label.pack(pady=(40, 30))

# מסגרת לכפתורים
frame = tk.Frame(window, bg="#1a1a2e")
frame.pack()

# כפתור Black Move
black_btn = tk.Button(
    frame,
    text="⬛  Black Move",
    command=black_move,
    bg="#111111",
    fg="#ffffff",
    font=("Georgia", 13, "bold"),
    width=14,
    height=2,
    relief="flat",
    cursor="hand2",
    activebackground="#333333",
    activeforeground="#ffffff"
)
black_btn.grid(row=0, column=0, padx=15)
black_btn.bind("<Enter>", on_enter_black)
black_btn.bind("<Leave>", on_leave_black)

# כפתור White Move
white_btn = tk.Button(
    frame,
    text="⬜  White Move",
    command=white_move,
    bg="#ffffff",
    fg="#111111",
    font=("Georgia", 13, "bold"),
    width=14,
    height=2,
    relief="flat",
    cursor="hand2",
    activebackground="#dddddd",
    activeforeground="#111111"
)
white_btn.grid(row=0, column=1, padx=15)
white_btn.bind("<Enter>", on_enter_white)
white_btn.bind("<Leave>", on_leave_white)

# שורת סטטוס
status = tk.Label(
    window,
    text="Select a move to begin",
    bg="#1a1a2e",
    fg="#666688",
    font=("Georgia", 10, "italic")
)
status.pack(pady=(30, 0))

window.mainloop()