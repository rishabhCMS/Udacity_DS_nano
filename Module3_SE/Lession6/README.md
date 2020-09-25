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
    
**Flask Basics**
