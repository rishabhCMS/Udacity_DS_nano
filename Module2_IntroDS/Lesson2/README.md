
## Communicating to the Stakeholders

### What to involve in a Readme?

    1. Installations
    2. Project Motivation
    3. FIle Descriptions
    4. How to Interact with the project
    5. Licensing, Authors, Acknowledgements, etc.
  
### Steps for creating a great blog post to cummunicate the results of your analysis

  1. Catch their eye: 
  
    a. captivating image
    b. supporting title
    c. connect with them, give a personal story they can relate to
    d. content throughout the page should be broken into 3 lines per idea
    e. keep sentences short and consize and ieas crisp and clear
    f. summarize and call the action (what do you think?)
    g. 8 min long read
    h. 200-250 words
    i outline woulb the questions you want to address
    j. view your work
      
      
### writing a good commit message

    feat: a new feature
    fix: a bug fix
    docs: changes to documentation
    style: formatting, missing semi colons, etc; no code change
    refactor: * refactoring production code
    test: adding tests, refactoring test; no production code change
    chore: updating build tasks, package manager configs, etc; no production code change

    https://udacity.github.io/git-styleguide/
    subject:
    
    Body:
    
    Footer:

### Contributing to open source projects

    https://blog.udacity.com/2013/10/get-started-with-open-source-projects.html
    
### tricks learned

````python
np.intersect1d(arr1, arr2) # finds intersection of two arrays
#Return the sorted, unique values that are in both of the input arrays.

set(arr1).intersection(arr2)
````


### Documentation

DOCUMENTATION: additional text or illustrated information that comes with or is embedded in the code of software.

Helpful for clarifying complex parts of code, making your code easier to navigate, and quickly conveying how and why different components of your program are used.

Several types of documentation can be added at different levels of your program:

    In-line Comments - line level
    Docstrings - module and function level
    Project Documentation - project level
    
[docstrings](https://www.python.org/dev/peps/pep-0257/)
[Project Documentation](https://github.com/twbs/bootstrap)    
    
    
### Version Control

#### Scenario 1

Let's walk through the git commands that go along with each step in the scenario you just observed in the video above.

STEP 1: You have a local version of this repository on your laptop, and to get the latest stable version, you pull from the develop branch.
Switch to the develop branch

    git checkout develop

Pull latest changes in the develop branch
    
    git pull

STEP 2: When you start working on this demographic feature, you create a new branch for this called demographic, and start working on your code in this branch.
Create and switch to new branch called demographic from develop branch

    git checkout -b demographic

Work on this new feature and commit as you go

    git commit -m 'added gender recommendations'
    git commit -m 'added location specific recommendations'
    ...

STEP 3: However, in the middle of your work, you need to work on another feature. So you commit your changes on this demographic branch, and switch back to the develop branch.

Commit changes before switching

    git commit -m 'refactored demographic gender and location recommendations '

Switch to the develop branch

    git checkout develop

STEP 4: From this stable develop branch, you create another branch for a new feature called friend_groups.
Create and switch to new branch called friend_groups from develop branch
    
    git checkout -b friend_groups

STEP 5: After you finish your work on the friend_groups branch, you commit your changes, switch back to the development branch, merge it back to the develop branch, and push this to the remote repository’s develop branch.

Commit changes before switching

    git commit -m 'finalized friend_groups recommendations '

Switch to the develop branch

    git checkout develop

Merge friend_groups branch to develop   

    git merge --no-ff friends_groups

Push to remote repository

    git push origin develop

STEP 6: Now, you can switch back to the demographic branch to continue your progress on that feature.
    
    Switch to the demographic branch
    git checkout demographic
        
        
#### Scenario 2

Let's walk through the git commands that go along with each step in the scenario you just observed in the video above.

Step 1: You check your commit history, seeing messages of the changes you made and how well it performed.
View log history
    
    git log

Step 2: The model at this commit seemed to score the highest, so you decide to take a look.
   
    Checkout a commit
    git checkout bc90f2cbc9dc4e802b46e7a153aa106dc9a88560

After inspecting your code, you realize what modifications made this perform well, and use those for your model.

Step 3: Now, you’re pretty confident merging this back into the development branch, and pushing the updated recommendation engine.
Switch to develop branch

    git checkout develop

Merge friend_groups branch to develop
    
    git merge --no-ff friend_groups

Push changes to remote repository
    
    git push origin develop
    
#### Scenario 3

Let's walk through the git commands that go along with each step in the scenario you just observed in the video above.

Step 1: Andrew commits his changes to the documentation branch, switches to the development branch, and pulls down the latest changes from the cloud on this development branch, including the change I merged previously for the friends group feature.
Commit changes on documentation branch

    git commit -m "standardized all docstrings in process.py"

Switch to develop branch

    git checkout develop

Pull latest changes on develop down

    git pull

Step 2: Then, Andrew merges his documentation branch on the develop branch on his local repository, and then pushes his changes up to update the develop branch on the remote repository.
Merge documentation branch to develop

    git merge --no-ff documentation

Push changes up to remote repository

    git push origin develop

Step 3: After the team reviewed both of your work, they merge the updates from the development branch to the master branch. Now they push the changes to the master branch on the remote repository. These changes are now in production.
Merge develop to master

    git merge --no-ff develop

Push changes up to remote repository

    git push origin master

#### resources

[Merge Conflicts](https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/about-merge-conflicts)

[Git Branching Strategy](https://nvie.com/posts/a-successful-git-branching-model/)

[Versioning in DS](https://shuaiw.github.io/2017/07/30/versioning-data-science.html)

[Version Control Production ML models](https://blog.algorithmia.com/how-to-version-control-your-production-machine-learning-models/)


# B. Software Engineering Part II

#### Testing

befor the code can be deployed it needs to be tested

#### [Unit tests](https://www.fullstackpython.com/integration-testing.html)

#### Unit testing tool

1. pytest
    
        To install pytest, run pip install -U pytest in your terminal. You can see more information on getting started here.

        Create a test file starting with test_
        Define unit test functions that start with test_ inside the test file
        Enter pytest into your terminal in the directory of your test file and it will detect these tests for you!
        test_ is the default - if you wish to change this, you can learn how to in this pytest configuration

        In the test output, periods represent successful unit tests and F's represent failed unit tests. Since all you see is what test functions failed, it's wise to have only one assert statement per test. Otherwise, you wouldn't know exactly how many tests failed, and which tests failed.

        Your tests won't be stopped by failed assert statements, but it will stop if you have syntax errors.
        


