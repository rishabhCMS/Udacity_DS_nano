# Lesson 4: Introduction to Object Oriented Programming (OOP)

## Why OOP?

    1. OOP allows us to create large modular programs that can expand over time

    2. OOP hides the implementation from the end-user.
    
[Source Repo](https://github.com/udacity/DSND_Term2/tree/master/lessons/ObjectOrientedProgramming/IdeFiles)

    objects are defined by characteristics and actions.
    
    for example: a colthing store is an object, the sales-person is an object, the cloth is an object
    
    here, if sales person is an object, they have the charateristics of name, age, hourly pay, hours worked, and her actions are to sell
    
## Class, Object, Method & Attribute

**Charateristics**: Attributes

**Actions**: Methods

    Difference between a method and a function is that, a function is defined outside the class and a method is defined inside the class
    
    ````python Self ```` tells Python where to look in the computer's memory for the object

    
### Setter and Getter methods in python

**Setter**: it is a method used to set the value of an attribute

**Getter**: It is a method used to get the value of an attribute

````python
# example
class Shirt():

    def __init__(self, shirt_color, shirt_size, shirt_price):
        self.color = shirt_color
        self.size = shirt_size
        self.price = shirt_price
        
    def get_price(self):   # getter function
        return(self.price)
        
    def set_price(self, new_price):    # setter functio
        self.price = new_price
````

### [Commenting Object Oriented Code](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

**Docstring**:A docstring is a type of comment that describes how a Python module, function, class or method works
 
 example with comments
 ````python
 class Pants:
    """The Pants class represents an article of clothing sold in a store
    """

    def __init__(self, color, waist_size, length, price):
        """Method for initializing a Pants object

        Args: 
            color (str)
            waist_size (int)
            length (int)
            price (float)

        Attributes:
            color (str): color of a pants object
            waist_size (str): waist size of a pants object
            length (str): length of a pants object
            price (float): price of a pants object
        """

        self.color = color
        self.waist_size = waist_size
        self.length = length
        self.price = price

    def change_price(self, new_price):
        """The change_price method changes the price attribute of a pants object

        Args: 
            new_price (float): the new price of the pants object

        Returns: None

        """
        self.price = new_price

    def discount(self, percentage):
        """The discount method outputs a discounted price of a pants object

        Args:
            percentage (float): a decimal representing the amount to discount

        Returns:
            float: the discounted price
        """
        return self.price * (1 - percentage)
 ````
## Gaussian Class

## Magic Methods

````python
#this fucntion is a magic function which is a reprenstation function for an object of that class 
def __repr__(self):

    """Function to output the characteristics of the Gaussian instance

    Args:
        None

    Returns:
        string: characteristics of the Gaussian

    """

    return "mean {}, standard deviation {}".format(self.mean, self.stdev)
````

## Inheritence

**Sample Inherience code**

````python
class Distribution:
    
    def __init__(self, mu=0, sigma=1):
    
        """ Generic distribution class for calculating and 
        visualizing a probability distribution.
    
        Attributes:
            mean (float) representing the mean value of the distribution
            stdev (float) representing the standard deviation of the distribution
            data_list (list of floats) a list of floats extracted from the data file
            """
        
        self.mean = mu
        self.stdev = sigma
        self.data = []


    def read_data_file(self, file_name):
    
        """Function to read in data from a txt file. The txt file should have
        one number (float) per line. The numbers are stored in the data attribute.
                
        Args:
            file_name (string): name of a file to read from
        
        Returns:
            None
        
        """
            
        with open(file_name) as file:
            data_list = []
            line = file.readline()
            while line:
                data_list.append(int(line))
                line = file.readline()
        file.close()
    
        self.data = data_list
````

````python
import math
import matplotlib.pyplot as plt

class Gaussian(Distribution):
    """ Gaussian distribution class for calculating and 
    visualizing a Gaussian distribution.
    
    Attributes:
        mean (float) representing the mean value of the distribution
        stdev (float) representing the standard deviation of the distribution
        data_list (list of floats) a list of floats extracted from the data file
            
    """
    def __init__(self, mu=0, sigma=1):
        
        Distribution.__init__(self, mu, sigma)
    
        
    
    def calculate_mean(self):
    
        """Function to calculate the mean of the data set.
        
        Args: 
            None
        
        Returns: 
            float: mean of the data set
    
        """
                    
        avg = 1.0 * sum(self.data) / len(self.data)
        
        self.mean = avg
        
        return self.mean



    def calculate_stdev(self, sample=True):

        """Function to calculate the standard deviation of the data set.
        
        Args: 
            sample (bool): whether the data represents a sample or population
        
        Returns: 
            float: standard deviation of the data set
    
        """

        if sample:
            n = len(self.data) - 1
        else:
            n = len(self.data)
    
        mean = self.calculate_mean()
    
        sigma = 0
    
        for d in self.data:
            sigma += (d - mean) ** 2
        
        sigma = math.sqrt(sigma / n)
    
        self.stdev = sigma
        
        return self.stdev
        
        
        
    def plot_histogram(self):
        """Function to output a histogram of the instance variable data using 
        matplotlib pyplot library.
        
        Args:
            None
            
        Returns:
            None
        """
        plt.hist(self.data)
        plt.title('Histogram of Data')
        plt.xlabel('data')
        plt.ylabel('count')
        
        
        
    def pdf(self, x):
        """Probability density function calculator for the gaussian distribution.
        
        Args:
            x (float): point for calculating the probability density function
            
        
        Returns:
            float: probability density function output
        """
        
        return (1.0 / (self.stdev * math.sqrt(2*math.pi))) * math.exp(-0.5*((x - self.mean) / self.stdev) ** 2)
        

    def plot_histogram_pdf(self, n_spaces = 50):

        """Function to plot the normalized histogram of the data and a plot of the 
        probability density function along the same range
        
        Args:
            n_spaces (int): number of data points 
        
        Returns:
            list: x values for the pdf plot
            list: y values for the pdf plot
            
        """
        
        mu = self.mean
        sigma = self.stdev

        min_range = min(self.data)
        max_range = max(self.data)
        
         # calculates the interval between x values
        interval = 1.0 * (max_range - min_range) / n_spaces

        x = []
        y = []
        
        # calculate the x values to visualize
        for i in range(n_spaces):
            tmp = min_range + interval*i
            x.append(tmp)
            y.append(self.pdf(tmp))

        # make the plots
        fig, axes = plt.subplots(2,sharex=True)
        fig.subplots_adjust(hspace=.5)
        axes[0].hist(self.data, density=True)
        axes[0].set_title('Normed Histogram of Data')
        axes[0].set_ylabel('Density')

        axes[1].plot(x, y)
        axes[1].set_title('Normal Distribution for \n Sample Mean and Sample Standard Deviation')
        axes[0].set_ylabel('Density')
        plt.show()

        return x, y
        
    def __add__(self, other):
        
        """Function to add together two Gaussian distributions
        
        Args:
            other (Gaussian): Gaussian instance
            
        Returns:
            Gaussian: Gaussian distribution
            
        """
        
        result = Gaussian()
        result.mean = self.mean + other.mean
        result.stdev = math.sqrt(self.stdev ** 2 + other.stdev ** 2)
        
        return result
        
        
    def __repr__(self):
    
        """Function to output the characteristics of the Gaussian instance
        
        Args:
            None
        
        Returns:
            string: characteristics of the Gaussian
        
        """
        
        return "mean {}, standard deviation {}".format(self.mean, self.stdev)
````
## Advance OOP Topics

1. [Class methods, instance methods and inheritence methods](https://realpython.com/instance-class-and-static-methods-demystified/)

2. [class attributes vs instance attributes0](https://www.python-course.eu/python3_class_and_instance_attributes.php)

3. [multiple inheritance, mixins](https://easyaspython.com/mixins-for-fun-and-profit-cb9962760556)

4. [Python decorators ](https://realpython.com/primer-on-python-decorators/)
