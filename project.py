import sympy as sp
import scipy.optimize as opt

class RootSolver:
    """
    A base class to represent root-solving methods.
    Each specific method (Newton's, Bisection, etc.) will inherit this class.
    """
    def __init__(self, func):
        """
        Initialize with the function to be solved.
        
        :param func: Function to find the root of (should be callable).
        """
        self.func = func

    def solve(self):
        """
        This method should be implemented by subclasses to solve the root-finding problem.
        """
        pass

class NewtonMethod(RootSolver):
    """
    Class implementing the Newton-Raphson method for root-finding.
    """
    def __init__(self, func, derivative, x0, tol=1e-6, max_iter=100):
        """
        Initialize the Newton's method solver.
        
        :param func: The function whose root is to be found.
        :param derivative: The derivative of the function.
        :param x0: Initial guess for the root.
        :param tol: Tolerance for the solution (default 1e-6).
        :param max_iter: Maximum number of iterations (default 100).
        """
        super().__init__(func)
        self.derivative = derivative
        self.x0 = x0
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        """
        Solve for the root using Newton's method.
        
        :return: The root of the function.
        """
        x = self.x0
        for _ in range(self.max_iter):
            fx = self.func(x)  # Evaluate the function at x
            fx_prime = self.derivative(x)  # Evaluate the derivative at x
            if abs(fx) < self.tol:  # Check if the function value is close enough to zero
                return x
            x = x - fx / fx_prime  # Newton-Raphson iteration
        return x  # Return the last value if max iterations are reached


class BisectionMethod(RootSolver):
    """
    Class implementing the Bisection method for root-finding.
    """
    def __init__(self, func, a, b, tol=1e-6, max_iter=100):
        """
        Initialize the Bisection method solver.
        
        :param func: The function whose root is to be found.
        :param a: Left interval endpoint.
        :param b: Right interval endpoint.
        :param tol: Tolerance for the solution (default 1e-6).
        :param max_iter: Maximum number of iterations (default 100).
        """
        super().__init__(func)
        self.a = a
        self.b = b
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        """
        Solve for the root using the Bisection method.
        
        :return: The root of the function.
        """
        if self.func(self.a) * self.func(self.b) >= 0:
            raise ValueError("Function has the same signs at the endpoints.")  # Check if the function changes sign
        
        for _ in range(self.max_iter):
            c = (self.a + self.b) / 2  # Midpoint
            if abs(self.func(c)) < self.tol:  # Check if the function value at c is close to zero
                return c
            elif self.func(c) * self.func(self.a) < 0:  # Root is between a and c
                self.b = c
            else:  # Root is between c and b
                self.a = c
        return (self.a + self.b) / 2  # Return the midpoint after max iterations
    
class SecantMethod(RootSolver):
    """
    Class implementing the Secant method for root-finding.
    """
    def __init__(self, func, x0, x1, tol=1e-6, max_iter=100):
        """
        Initialize the Secant method solver.
        
        :param func: The function whose root is to be found.
        :param x0: First initial guess.
        :param x1: Second initial guess.
        :param tol: Tolerance for the solution (default 1e-6).
        :param max_iter: Maximum number of iterations (default 100).
        """
        super().__init__(func)
        self.x0 = x0
        self.x1 = x1
        self.tol = tol
        self.max_iter = max_iter

    def solve(self):
        """
        Solve for the root using the Secant method.
        
        :return: The root of the function.
        """
        for _ in range(self.max_iter):
            fx0 = self.func(self.x0)  # Evaluate the function at x0
            fx1 = self.func(self.x1)  # Evaluate the function at x1
            if abs(fx1) < self.tol:  # If the value of the function is small enough
                return self.x1
            # Secant method iteration
            x2 = self.x1 - fx1 * (self.x1 - self.x0) / (fx1 - fx0)
            self.x0, self.x1 = self.x1, x2
        return self.x1  # Return the last value if max iterations are reached



class FsolveMethod(RootSolver):
    """
    Class using the fsolve method from scipy for root-finding.
    """
    def __init__(self, func, guess):
        """
        Initialize the fsolve method solver.
        
        :param func: The function whose root is to be found.
        :param guess: Initial guess for the root.
        """
        super().__init__(func)
        self.guess = guess

    def solve(self):
        """
        Solve for the root using the fsolve method from SciPy.
        
        :return: The root of the function.
        """
        return opt.fsolve(self.func, self.guess)  # Use fsolve to find the root




class EquationParser:
    """
    Class for parsing an equation given as a string and converting it into a callable function.
    """
    def __init__(self, equation_str):
        """
        Initialize with the equation string.
        
        :param equation_str: String representing the equation (use 'x' as variable).
        """
        self.equation_str = equation_str
    
    def parse(self):
        """
        Parse the equation string and return a callable function.
        
        :return: A callable function representing the equation.
        """
        x = sp.symbols('x')
        equation = sp.sympify(self.equation_str)  # Convert string to sympy expression
        return sp.lambdify(x, equation, 'numpy')  # Convert sympy expression to callable function using numpy

class UserInterface:
    """
    Class that handles user interaction to choose the solving method and input the equation.
    """
    def __init__(self):
        self.method = None  # This will store the chosen solving method
        self.func = None  # This will store the callable function representing the equation

    def get_user_input(self):
        """
        Get user input for the equation and solving method.
        
        :return: The user's choice of method.
        """
        print("Welcome to the Nonlinear Equation Solver!")
        
        # Input the equation as a string
        equation = input("Enter the equation to solve (use 'x' as the variable): ")
        
        # Input the method choice from the user
        method_choice = input("Choose a method (1 = Newton's Method, 2 = Bisection, 3 = Secant, 4 = fsolve): ")
        
        # Parse the equation into a callable function
        parser = EquationParser(equation)
        self.func = parser.parse()
        
        return method_choice

    def choose_method(self):
        """
        Choose the root-finding method based on user input.
        """
        method_choice = self.get_user_input()  # Get user's choice of method
        
        # Choose the method based on user's input
        if method_choice == '1':  # Newton's Method
            x0 = float(input("Enter the initial guess: "))
            derivative_str = input("Enter the derivative of the function: ")
            derivative = EquationParser(derivative_str).parse()  # Parse the derivative as well
            self.method = NewtonMethod(self.func, derivative, x0)
        
        elif method_choice == '2':  # Bisection Method
            a = float(input("Enter the left interval a: "))
            b = float(input("Enter the right interval b: "))
            self.method = BisectionMethod(self.func, a, b)
        
        elif method_choice == '3':  # Secant Method
            x0 = float(input("Enter the first initial guess: "))
            x1 = float(input("Enter the second initial guess: "))
            self.method = SecantMethod(self.func, x0, x1)
        
        elif method_choice == '4':  # fsolve Method
            guess = float(input("Enter the initial guess: "))
            self.method = FsolveMethod(self.func, guess)
        
        else:
            print("Invalid method choice. Please select a valid option.")
            return

    def solve(self):
        """
        Solve the equation using the selected method and print the result.
        """
        if self.method:
            root = self.method.solve()  # Solve the equation using the chosen method
            print(f"Root of the equation: {root}")
        else:
            print("No method selected. Please choose a method first.")
            
# Example of how the program can be run
if __name__ == "__main__":
    ui = UserInterface()  # Create a UserInterface object
    ui.choose_method()  # Prompt the user to choose a method and input parameters
    ui.solve()  # Solve the equation using the chosen method

