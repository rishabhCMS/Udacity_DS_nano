# Lesson 6: Web development

## Lesson outline

Basics of a web app
  - html
  - css  
  - javascript

Front-end libraries
  - boostrap
  - plotly

Back-end libraries
  - flask
  - Deploy a web app to the cloud
  
[Resource repo](https://github.com/udacity/DSND_Term2/tree/master/lessons/WebDevelopment)

## Components of a web app

A. Front end: HTML, CSS, JS, bootstrap(to prganize the front -end)

B. Backend: server where all the HTML, CSS, JS files needed to load the webpage are stored. languages used are ruby, Java, python

## Front End
 
example html code
```html
<!DOCTYPE html>

<html>

<head>
    <title>Page Title</title>
</head>

<body>
    <h1>A Photo of a Beautiful Landscape</h1>
    <a href="https://www.w3schools.com/tags">HTML tags</a>
    <p>Here is the photo</p>
    <img src="photo.jpg" alt="Country Landscape">
</body>

</html>
```

**important Links**

[W3Schools HTML Tags](https://www.w3schools.com/tags/default.asp)

[W3C Validator](https://validator.w3.org/#validate_by_input)


## ids and classes

```html
<div id="top">
    <p class="first_paragraph">First paragraph of the section</p>
    <p class="second_paragraph">Second paragraph of the section</p>
</div>

<div id="bottom">
    <p class="first_paragraph">First paragraph of the section</p>
    <p class="second_paragraph">Second paragraph of the section</p>
</div>
```

## CSS
 
Cascading style Sheets, we write CSS to style each element of the HTML

There are two ways to implement CSS

    1. inline CSS : you add style rules inside the tags
    2. CSS wrtten in a style sheet
    
We'll be using the [bootstrap](https://getbootstrap.com/) library for wrting CSS

inline CSS
```html
<p style="font-size:20px;">This is a paragraph</p>
```

Style Sheet
```html
...
<head>
   <style>
       p {font-size: 20px;}
   </style>
</head>
```

Or CSS can g ina separate sheet
```html
<head>
    <link rel="stylesheet" type"text/css" href="style.css">
</head>
```

[CSS rules](https://www.w3schools.com/css/default.asp)

The general syntax is that you:

    - select the html element, id, and/or class of interest
    - specify what you want to change about the element
    - specify a value, followed by a semi-colon
    
## [Bootstap library](https://getbootstrap.com/)

Bootstrap is one of the easier front-end frameworks to work with. Bootstrap eliminates the need to write CSS or JavaScript. Instead, you can style your websites with HTML. You'll be able to design sleek, modern looking websites more quickly than if you were coding the CSS and JavaScript directly.

- [Starter Template](https://getbootstrap.com/docs/4.0/getting-started/introduction/#starter-template)
- [Column Grid Explanation](https://getbootstrap.com/docs/4.0/layout/grid/)
- [Containers and Responsive Layout](https://getbootstrap.com/docs/4.0/layout/overview/)
- [Images](https://getbootstrap.com/docs/4.0/content/images/)
- [Navigation Bars](https://getbootstrap.com/docs/4.0/components/navbar/)
- [Font Colors](https://getbootstrap.com/docs/4.0/utilities/colors/)

[bootstrap example](https://youtu.be/KsrqjguHWUI)

## Javascript


## the backend

## Flask

it is wrtten in python

**Steps involved**

    - setting up the backend
    - linking the backend and the frontend together
    - deploying the app to a server so that the app is available from a web address
    
**What is [Flask](https://flask.palletsprojects.com/en/1.1.x/)?**

abstraction is basically hiding the un-necessary details from the programer
for ex: if I want to make a cup of tea, all I need is tea bag, a mug, and hot water. I don't want to make/know about how to make a mug from clay, I don't want to know how the electric kettle was built.

Similarly Flask abstracts the code for recieving requests, interpreting the requests and sending out correct files.

if I want to access website
    
      1. I go to my browser, type in the http request
      2. the browser sends out the request to the server
      3. Flask is a software at the server, which recieves the request, interprets it and sends out the relevant info
      4. the browser recieves the http message from the server and displays it
      
**Why work with Flask?**

    1. it's wrtten in python
    2. it is easy to use for making a smal web app
    3. bc Flask is written in python you can use flask with any python lib
    
**[Flask Basics](https://youtu.be/i_U3O-7cymk)**

**1. Using Flask in classroom work space

````console
python worldbank.py
````
**2. Seeing your app in the workspace**

````console
env | grep WORK
````

````console
output:
WORKSPACEDOMAIN=udacity-student-workspaces.com
WORKSPACEID=viewc7f3319f2

https://viewc7f3319f2-3001.udacity-student-workspaces.com/
````

**3. Creating new web-pages **

To create a new web page, you first need to specify the route in the routes.py as well as the name of the html template.
````python
@app.route('/new-route')
def render_the_route():
    return render_template('new_route.html')
````
The route name, function name, and template name do not have to match; however, it's good practice to make them similar so that the code is easier to follow.

The new_route.html file must go in the templates folder. Flask automatically looks for html files in the templates folder.



**What is @app.route?**
To use Flask, you don't necessarily need to know what @app.route is doing. You only have to remember that the path you place inside of @app.route() will be the web address. And then the function you write below @app.route is used to render the correct html template file for the web address.

In Python, the @ symbol is used for decorators. Decorators are a shorthand way to input a function into another function. Take a look at this code. Python allows you to use a function as an input to another function:

````python
def decorator(input_function):

    return input_function

def input_function():
    print("I am an input function")

decorator_example = decorator(input_function)
decorator_example()
````
Running this code will print the string:

`I am an input function`

Decorators provide a short-hand way of getting the same behavior:
````python
def decorator(input_function):
    print("Decorator function")
    return input_function

@decorator
def input_function():
    print("I am an input function")

input_function()
````
This code will print out:

`Decorator function
I am an input function`

Instead of using a decorator function, you could get the same behavior with the following code:
````python
input_function = decorator(input_function)
input_function()
````
Because `@app.route()` has the `.`symbol, there's an implication that app is a class (or an instance of a class) and route is a method of that class. Hence a function written underneath `@app.route()` is going to get passed into the `route` method. The purpose of `@app.route()` is to make sure the correct web address gets associated with the correct html template. This code
````python
@app.route('/homepage')
def some_function()
  return render_template('index.html')
````
is ensuring that the web address `www.website.com/homepage` is associated with the `index.html` template.

If you'd like to know more details about decorators and how `@app.route()` works, check out these tutorials:

[how @app.route works](https://ains.co/blog/things-which-arent-magic-flask-part-1.html)

[general decorators tutorial](https://realpython.com/primer-on-python-decorators/)

## Flask + pandas

Since Flask is written in python you can use it with varois python libraries

## Flask + Plotly + pandas 

[vid 1](https://youtu.be/xg7P8MnItdI)

    Step 1: Wrangle the data using Pandas.
    Step 2: Visulize the data using plotly
    Step 3: Show the visulaization on the front-end
    
### Part I:Passing data-set from back-end-> front-end

The purpose of this section is to give you an idea of how the final web app works in terms of passing information back and forth between the back end and front end. The web template you'll be using at the end of the lesson will already provide the code for sharing information between the back and front ends. Your task will be to wrangle data and set up the plotly visualizations using Python. But it's important to get a sense for how the web app works.

In the video above, the data set was sent from the back end to the front end. This was accomplished by including a variable in the render_template() function like so:

````python
data = data_wrangling()

@app.route('/')
@app.route('/index')
def index():
   return render_template('index.html', data_set = data)
````
What this code does is to first load the data using the data_wrangling function from wrangling.py. This data gets stored in a variable called data.

In render_template, that data is sent to the front end via a variable called data_set. Now the data is available to the front_end in the data_set variable.

In the index.html file, you can access the data_set variable using the following syntax:

`{{ data_set }}`
You can do this because Flask comes with a template engine called Jinja. Jinja also allows you to put control flow statements in your html using the following syntax:

````html
{% for tuple in data_set %}
  <p>{{tuple}}</p>
{% end_for %}
````
The logic is:

    1. Wrangle data in a file (aka Python module). In this case, the file is called wrangling.py. The wrangling.py has a function that returns the clean data.

    2. Execute this function in routes.py to get the data in routes.py
    
    3. Pass the data to the front-end (index.html file) using the render_template method.
    
    4. Inside of index.html, you can access the data variable with the squiggly bracket syntax `{{ }}`
    
### Part II: Create Plotly viz in back-end and send it to Front-end
[vid2](https://youtu.be/yx-DRzMsblI)

In the second part, a Plotly visualization was set up on the back-end inside the routes.py file using Plotly's Python library. The Python plotly code is a dictionary of dictionaries. The Python dictionary is then converted to a JSON format and sent to the front-end via the render_templates method.

Simultaneously a list of ids are created for the plots. This information is also sent to the front-end using the render_template() method.

On the front-end, the ids and visualization code (JSON code) is then used with the Plotly javascript library to render the plots.

In summary:

    - Python is used to set up a Plotly visualization
    
    - An id is created associated with each visualization
    
    - The Python Plotly code is converted to JSON
    
    - The ids and JSON are sent to the front end (index.html).
    
    - The front end then uses the ids, JSON, and JavaScript Plotly library to render the plots.
    
### Part III: Making complex visualizations in Plotly

[vid3](https://youtu.be/e8owK5zk-g8)

### Part VI: add more visualizations in the back end code and then render those visualizations on the front end

[vid4](https://youtu.be/4IF2G9Fehb4)

Beyond a CSV file
Besides storing data in a local csv file (or text, json, etc.), you could also store the data in a database such as a SQL database.

The database could be local to your website meaning that the database file is stored on the same server as your website; alternatively, the database could be stored somewhere else like on a separate database server or with a cloud service like Amazon AWS.

Using a database with your web app goes beyond the scope of this introduction to web development, here are a few resources for using databases with Flask apps:

[Tutorial](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-iv-database) - Using Databases with Flask
[SQL Alchemy](http://docs.sqlalchemy.org/en/latest/)- a Python toolkit for working with SQL
[Flask SQLAlchemy](http://flask-sqlalchemy.pocoo.org/2.3/) - a Flask library for using SQLAlchemy with Flask

## [Deployment](https://youtu.be/YPfNzpnm_Rk)

Other Services Besides Heroku
Heroku is just one option of many for deploying a web app, and Heroku is actually owned by Salesforce.com.

The big internet companies offer similar services like Amazon's Lightsail, Microsoft's Azure, Google Cloud, and IBM Cloud (formerly IBM Bluemix). However, these services tend to require more configuration. Most of these also come with either a free tier or a limited free tier that expires after a certain amount of time.

Instructions Deploying from the Classroom
Here is the code used in the screencast to get the web app running:

First, a new folder was created for the web app and all of the web app folders and files were moved into the folder:
````console
mkdir web_app
mv -t web_app data worldbankapp wrangling_scripts worldbank.py
````
The next step was to create a virtual environment and then activate the environment:
````console
conda update python
python3 -m venv worldbankvenv
source worldbankenv/bin/activate
````
Then, pip install the Python libraries needed for the web app

`pip install flask pandas plotly gunicorn`
The next step was to install the heroku command line tools:
````console
curl https://cli-assets.heroku.com/install-ubuntu.sh | sh
https://devcenter.heroku.com/articles/heroku-cli#standalone-installation
heroku —-version
````
And then log into heroku with the following command

heroku login
Heroku asks for your account email address and password, which you type into the terminal and press enter.

The next steps involved some housekeeping:

`remove app.run()` from `worldbank.py`
type `cd web_app` into the Terminal so that you are inside the folder with your web app code.
Then create a `proc` file, which tells Heroku what to do when starting your web app:

`touch Procfile`
Then open the Procfile and type:

`web gunicorn worldbank:app`
Next, create a requirements file, which lists all of the Python library that your app depends on:

`pip freeze > requirements.txt`
And initialize a git repository and make a commit:
````console
git init
git add .
git commit -m ‘first commit’
````
Now, create a heroku app:

`heroku create my-app-name`
where `my-app-name` is a `unique name` that nobody else on Heroku has already used.

The heroku create command should create a git repository on Heroku and a web address for accessing your web app. You can check that a remote repository was added to your git repository with the following terminal command:

`git remote -v`
Next, you need to push your git repository to the remote heroku repository with this command:

`git push heroku master`
Now, you can type your web app's address in the browser to see the results.

## Virtual Environments vs. Anaconda
Virtual environments and Anaconda serve a very similar purpose. Anaconda is a distribution of Python (and the analytics language R) specifically for data science. Anaconda comes installed with a package and environment manager called conda. You can create separate environments using conda. However, these environments automatically come with Python packages meant for data science.

Virtual environments, on the other hand, come with the Python language but do not pre-install other packages.

The classroom workspace has many other Python libraries pre-installed including an installation of Anaconda.

When installing a web app to a server, you should only include the packages that are necessary for running your web app. Otherwise you'd be installing Python packages that you don't need.

To ensure that your app only installs necessary packages, you should create a virtual Python environment. A virtual Python environment is a separate Python installation on your computer that you can easily remove and won't interfere with your main Python installation.

There is more than one Python package that can set up virtual environments. In the past, you had to install these packages yourself. With Python 3.6, there is a virtual environment package that comes with the Python installation. The packaged is called venv

However, there is a bug with anaconda's 3.6 Python installation on a Linux system. So in order to use venv in the workspace classroom, you first need to update the Python installation as shown in the instructions above.

Creating a Virtual Environment in the Classroom
Open a terminal window in a workspace and type:

conda update python
When asked for confirmation, type y and hit enter. Your Python installation should update.

Next, make sure you are in the folder where you want to build your web app. In the classroom, the workspace folder is fine. But on your personal computer, you'll want to make a new folder. For example:

mkdir myapp
will create a new folder called myapp and cd myapp will change your current directory so that you are inside the myapp folder.

Then to create a virtual environment type:

python3 -m venv name
where name can be anything you want. You'll see a new folder appear in the workspace with your environment name.

Finally, to activate the virtual environment. Type:

source name/bin/activate
You can tell that your environment is activated because the name will show up in parenthesis on the left side of the terminal.

Creating a Virtual Environment Locally on Your Computer
You can develop your app using the classroom workspace. If you decide to develop your app locally on your computer, you should set up a virtual environment there as well. Different versions of Python have different ways of setting up virtual environments. Assuming you are using Python 3.6 and are on a linux or macOS system, then you should be able to set up a virtual environment on your local machine just by typing:

python3 -m venv name
and then to activate:

source name/bin/activate
On Windows, the command is;

c:\>c:\Python35\python -m venv c:\path\to\myenv
and to activate:

C:\> <venv>\Scripts\activate.bat
For more information, read through this link.
