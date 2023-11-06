# User Interface
import tkinter as tk
from tkinter import *
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt


# Embbed Graph in UI
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import quad
import warnings

warnings.filterwarnings("ignore")
# Custom fourier library
import fourier

period = 0
N = 0
a_0 = 0
a_n = 0
b_n = 0


# Makes the function periodic
def periodic_extension(a, b):
    T = b - a
    return lambda f: lambda x: f((x - a) % T + a)


# Check if an entry has ""
def check_functions_is_null():
    equations = list(all_equations)  # Transform to array
    is_null = False

    for piece in equations:
        for value in piece.values():
            if value.get() == "":
                return True

    return False


# Plots the funcion and it's series
def plot_function():
    equations = list(all_equations)  # Transform to array

    if not check_functions_is_null():
        ax.clear()  # Clear the previous plot (if any)
        try:
            # Get the linspace intervals for the graph
            lower_linspace = equations[0]
            lower_linspace = float(lower_linspace["lower_bound"].get())
            upper_linspace = equations[-1]
            upper_linspace = float(upper_linspace["upper_bound"].get())

            t = np.linspace(lower_linspace, upper_linspace, 500)
            print("Ploting original function.")
            for equation_piece in equations:
                func = lambda t: eval(
                    equation_piece["equation"].get()
                )  # Transform string to lambda function
                lower_bound = float(equation_piece["lower_bound"].get())
                upper_bound = float(equation_piece["upper_bound"].get())

                t_interval = np.linspace(lower_bound, upper_bound, 500)
                y_interval = [func(t) for t in t_interval]

                # Plot Function
                ax.plot(
                    t_interval,
                    y_interval,
                    label="$f(t)$",
                    color="blue",
                    linestyle="dashed",
                )

            T = upper_linspace - lower_linspace
            period_result.set(f"{T}s")

            N = fourier.ice(equations, T)
            n_result.set(f"{N}")
            print("Calculating Series.")

            # Calculate Fourier Series
            @periodic_extension(lower_linspace, upper_linspace)
            def get_equation(x):
                for function in equations:
                    func = lambda t: eval(function["equation"].get())
                    lower_bound = float(function["lower_bound"].get())
                    upper_bound = float(function["upper_bound"].get())

                    if lower_bound <= x < upper_bound:
                        return func(x)
                return 0

            s_n = fourier.fourier_suma_parcial(get_equation, T, N)

            y_values_aprox = [s_n(x) for x in t]
            ax.plot(t, y_values_aprox, label="Approximation", color="red")

            a_0 = fourier.symbolic_a0(equations, T)
            a_n = fourier.symbolic_an(equations, T)  # .args[1][0]
            b_n = fourier.symbolic_bn(equations, T)  # .args[1][0]

            n_result.set(f"{N}")
            a0_result.set(f"{a_0}")
            an_result.set(f"{a_n}")
            bn_result.set(f"{b_n}")

            plt.grid(True)
            plt.legend()
            canvas.draw()

        except:
            messagebox.showerror("Error!", "The function has an error!")
    else:
        messagebox.showerror("Error!", "Please, fill all fields.")


# Make a pop-up window for instructions
def show_instructions():
    popup = tk.Toplevel(window)
    popup.title("Instructions")

    instructions = "Bienvenidos al programa\n\n"
    instructions += "1. Por favor usar funciones $f(t)$.\n"
    instructions += "2. Para funciones trigonometricas, exponenciales. Por favor usar 'np.sin(t)'.\n"
    instructions += "3. El tiempo de espera de la calculacion puede ser largo.\n"

    label = tk.Label(popup, text=instructions)
    label.pack(padx=20, pady=20)


# Adds more Functions
def addEquations():
    # Get number of next free row
    next_row = len(all_equations)

    # add label in next free row
    lab = Label(equation_info_frame, text=str(f"{next_row+1}: "))
    lab.grid(row=next_row + 1, column=0)

    # add entry in second row
    equation_piece = Entry(equation_info_frame)
    equation_piece.grid(row=next_row + 1, column=1)

    # add label in next free row
    lab = Label(equation_info_frame, text=",")
    lab.grid(row=next_row + 1, column=2)

    # add lower bound
    low_bound = Entry(equation_info_frame)
    low_bound.grid(row=next_row + 1, column=3)

    lab = Label(equation_info_frame, text="< t <")
    lab.grid(row=next_row + 1, column=4)

    # add upper bound
    upper_bound = Entry(equation_info_frame)
    upper_bound.grid(row=next_row + 1, column=5)

    all_equations.append(
        {
            "equation": equation_piece,
            "lower_bound": low_bound,
            "upper_bound": upper_bound,
        }
    )


# Creates main window
window = tk.Tk()
# Names the tab
window.title("Serie de Fourier Truncada")

# Create frame for organization
frame = tk.Frame(window)
frame.pack()

# Create input frame for the equations
equation_info_frame = tk.LabelFrame(frame, text="Enter equations f(t)")
equation_info_frame.grid(row=0, column=0, padx=20, pady=20)

equation_label = tk.Label(equation_info_frame, text="Equations: ")
equation_label.grid(row=0, column=0)

all_equations = []
addBoxButton = Button(
    equation_info_frame,
    text="Add Piecewise",
    font=("Arial", 12),
    fg="Blue",
    command=addEquations,
)
addBoxButton.grid(row=0, column=1)

print_function = Button(
    equation_info_frame,
    text="Graph & Calculate",
    font=("Arial", 12, "bold"),
    fg="Red",
    command=plot_function,
)
print_function.grid(row=0, column=2)

show_instructions_button = Button(
    equation_info_frame,
    text="Instructions",
    font=("Arial", 12, "bold"),
    command=show_instructions,
)
show_instructions_button.grid(row=0, column=3)

###############################################################################################
# Create info frame for the series
series_info_frame = tk.LabelFrame(frame, text="Series Info")
series_info_frame.grid(row=0, column=1, padx=20, pady=20)

# Period Section
period_result_label = tk.Label(series_info_frame, text="T =")
period_result_label.grid(row=0, column=0)

period_result = StringVar()


period_result_readOnly = tk.Label(series_info_frame, textvariable=period_result)
period_result_readOnly.grid(row=0, column=1)

# N Section
n_result_label = tk.Label(series_info_frame, text="N = ")
n_result_label.grid(row=1, column=0)

n_result = StringVar()

n_result_readOnly = tk.Label(series_info_frame, textvariable=n_result)
n_result_readOnly.grid(row=1, column=1)

# Coefficients section
coefficient_title = tk.Label(series_info_frame, text="Series Coefficients")
coefficient_title.grid(row=2, column=0)

coefficient_title = tk.Label(series_info_frame, text="a_0 =")
coefficient_title.grid(row=3, column=0)

coefficient_title = tk.Label(series_info_frame, text="a_n =")
coefficient_title.grid(row=4, column=0)

coefficient_title = tk.Label(series_info_frame, text="b_n =")
coefficient_title.grid(row=5, column=0)

a0_result = StringVar()
an_result = StringVar()
bn_result = StringVar()

n_result_readOnly = tk.Label(series_info_frame, textvariable=a0_result)
n_result_readOnly.grid(row=3, column=1)

n_result_readOnly = tk.Label(series_info_frame, textvariable=an_result)
n_result_readOnly.grid(row=4, column=1)

n_result_readOnly = tk.Label(series_info_frame, textvariable=bn_result)
n_result_readOnly.grid(row=5, column=1)

###################################################################
# frame for graph
graph_frame = tk.LabelFrame(frame, text="Graph")
graph_frame.grid(row=1, column=0, padx=20, pady=20)

# Create a matplotlib figure and canvas
fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=graph_frame)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack()

# Run the UI
window.mainloop()
