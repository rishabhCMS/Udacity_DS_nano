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

