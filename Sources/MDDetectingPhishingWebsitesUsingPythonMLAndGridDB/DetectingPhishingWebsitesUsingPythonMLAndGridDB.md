# Use Machine Learning and GridDB to Detect Phishing Websites 

   Curiosity alone can lead to getting your personal information leaked to anyone out there willing to have it. Are you the type that just clicks on any link a friend sends to you? It's dangerous! Hackers have so many ways of gathering information - they can gain your trust by social engineering and make you to do the things you would not do if it were just asked by a normal friend, and one of them is making you click on links.

   Phishing is a great technique used by hackers to gather information - passwords, emails, name, et cetera. When the process is successful, it gets your device vulnerable to exploits from them and they just hack you (what they call pwning). Phishing comes with a lot of high chances because less work is done to gain the information.
    
   How is this done? One of the various ways the Facebook account can be hacked us by phishing and this has made a lot of users to lose their Facebook accounts due to that. The hacker clones the page where Facebook asks you to change your password or the page that asks you to login to Facebook. Then, he sets the backend to store anything you insert into a database they can access. When you change your password, you're typing your old password as well as the new password. However, you're informing the hacker, "hey, here's my password, do whatever you like with it" and he takes that as a great opportunity. After some time gaining your trust, probably on WhatsApp, he uses a tool to make the link have the format of a normal Facebook link and sends it to you, the test case (with an intriguing caption). You click on the link, fill in your details and he gains access to them that way. The next time you login to Facebook, you would notice that you can't access your account anymore. That's how phishing works! 
   
   Now, it is very imperative to have a method of detecting phishing websites. However, one could be safe by just being able to recognize phishing links. Below are flaws you could find in links to classify as being fake or not:
   * The link is shortened
   * The link is mispelt
   * You have probably not heard of the website before or it ranks low
   * It is very long, having a combination of legit and unoriginal links

Hitherto, if you get to the website, there are also some tips that would help you detect a phishing website.
   * You would notice that the right-click button of your mouse has been disabled on the website. Now, hackers are really wise and very careful at what they do. Disabling the right-click button of your mouse prevents you from viewing page source (the source code of the page) because this would make you see what they don't want you to see. Though, you may not understand the code especially if you're not a programmer but you can still read English, and that matters a lot, to them.
   * Getting regular pop-up windows can also be a way to detect a phishing website. A pop-up window, as the name implies is a mini window that pops up from the top of the page. When you get regular pop-ups, this should be a time to be suspicious about the website.

##### * This blog would guide you through how to build your own model to detect a phishing website.
* Here are the required tools for this tutorial:
    I would assume that you are using the Windows operating system, preferably Windows 7 or higher.
    * Python 3.8 or Python 3.9.
      Get the latest version of Python from [the website](https://www.python.org/downloads)
    * A text editor like Notepad, Notepad++, Vim, Sublime, et cetera.
      Notepad comes preinstalled on every PC but you would have to download all other text editors
    * Jupyter Notebook (optional, if you already have the IDLE for Python).
      This is an environment used to write Python codes and it provides suitable cells where you can write code or text.
      You can get the Notebook by downloading Anaconda3 from https://www.anaconda.org, this would install some packages on your PC, including Jupyter Notebook and Spyder.
        
 Now let's move!    
 

### How to install the required Python libraries using Windows CMD

The required Python libraries for this tutorial are:
   * GridDB Python Client - A great database tool for IoT (Internet of Things) with both NoSQL and MySQL interfaces (SQL - Structured Query Language). This would be our choice of database for our model. GridDB was written in C++ and the Python Client was written in 
   * NumPy (Numerical Python) - this is very useful if you want to deal with arrays and maths.
   * SciPy (Scientific Python) - similar to NumPy, provides various maths, science, engineering and optimization algorithms.
   * Matplotlib - an alternative to MatLab, provides various visualizations using Python.
   * Scikit-learn - sklearn for short. This is one of the most important libraries used in Machine Learning as it has a lot            of modules that can be used for suitable purposes.
   * Pandas - this is the key to opening every dataset you want to use in your Data Science or Machine Learning projects as              it allows you to read datasets, especially csv (Comma-separated values) files to use in our Machine Learning model.              It can also be used for data visualization but as of this blog, we would use Matplotlib and Seaborn.
   * Seaborn - this is also a data visualization library, similar to Matplotlib. It allows you to plot different graphs and              plots in your code to check correlations between features and all that. No worries, we will get to business.

Windows CMD comes preinstalled on every Windows PC and it is a very great tool that allows you to access a lot of things by typing commands. You could even make games with it. We are going to use this tool to download the libraries we would use in our Machine Learning model. Follow the steps below to install the libraries on the Python software you downloaded:
   * On Desktop, open your Start Menu and search for **RUN** or press Windows icon+R, open it and type **%localappdata%**, then      press enter.
   * A new window appears which has the directory as ../AppData/Local. This window cannot be accessed by just navigating from        your Local Disk folder as it is hidden by default.
   * Scroll down to **p** or press p on your keyboard and find the **Programs** folder.
   * Open the **Programs** folder, you would see another folder in a new tab which is the **Python** folder.
   * Open the **Python** folder and a new tab which shows a folder with Python and the version of the Python installed as the        name of the folder; open this folder.
   * Finally, when another new tab appears, find the **Scripts** folder and open it. Check if the **pip** module is installed;        if not, no worries. Just copy the address (path of the tab you are in) by right-clicking on the top address tab, and            selecting ***Copy address*** or an option similar.
   * Open CMD by searching for it as you did for the RUN window; right-click and select ***Run as administrator***.
   * A Black window appears, don't get scared. That's the Command Prompt window and we will be using that to install the Python      libraries we would use.
   * Type **cd <THE_ADDRESS_YOU_COPIED_EARLIER>** and press enter. For example, if the address you copied was                        ***C:\Users\hp\AppData\Local\Programs\Python\Python38-32\Scripts***, you would type **cd                                        C:\Users\hp\AppData\Local\Programs\Python\Python38-32\Scripts**. This would set the current directory to the Scripts folder      which has the PIP module we want to use.
   * If you don't have PIP installed on your PC already, visit [this page](https://bootstrap.pypa.io/get-pip.py) and download the page as you      would save a web page. This page would be downloaded as ***get-pip.py*** file. Open this file, another CMD window would          appear and this would install the latest PIP version on your Python software, which is pip21.0.1 (updated on Jan 30, 2021;      13:54:11) as of this writing.
   * When the installation is done, you can now use the CMD Window you ran as administrator ealier. Then, you can install the        modules using pip.    

#### Installation
Since you now have the PIP module installed. You can type PIP commands on your Command Prompt Window to install the libraries. Type the following commands and press enter after each command to install the modules:
   
   ***pip install griddb-python***
    
   ***pip install numpy***
    
   ***pip install pandas***
    
   ***pip install matplotlib***
    
   ***pip install scikit-learn***
    
   ***pip install seaborn***
    
   ***pip install scipy***
    

#### Now, let's get to the real business

The dataset we will be using for our model is the one provided by the University of California Irvine (UCI) website at https://archive.ics.uci.edu/ml/datasets/phishing+websites. This contains many numeric features that could help us classify websites as being legit or fake (real website or phishing website). The dataset has 2456 instances and 30 attributes as given by the website. It's always a good practice to check for missing values whenever we want to analyse data, though the website says there are no missing values. On the top of the page, you would see a Download label and two buttons; click on the first button which says **Data Folder**. It would redirect you to the page you can download the dataset (as an arff file). Click on the **Training Dataset.arff** file to download it. 

   However, you could decide to just visit [the UCI dataset page for it](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Training%20Dataset.arff) to download the dataset directly.
   
   To understand all the features, download the Phishing Website Features.docx file from [here](https://archive.ics.uci.edu/ml/machine-learning-databases/00327/Phishing%20Websites%20Features.docx). 

### Converting ARFF file to CSV

Since the dataset downloaded from the UCI website is an ARFF file, there's need to convert it into a CSV file so we can use it in our Python code. Make sure the arff dataset file you downloaded is in the same folder as the one containing your Python code.


```python
import glob as gb # This builtin module would allow us to select only the file with .arff extension

files = [f for f in gb.glob("*.arff")] # We are using "files" because this selects all the files with that extension

# This function would convert the ARFF file to CSV file
def convert(lines):
    header = ""
    file_content = []
    data = not True
    
    for line in lines:
        if not data:
            if "@attribute" in line:
                attributes = line.split()
                columnName = attributes[attributes.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += "\n"
                file_content.append(header)
        else:
            file_content.append(line)
    return file_content
        

for file in files:
    with open(file, "r") as inp:
        lines = inp.readlines()
        output = convert(lines)
        with open("dataset" + ".csv", "w") as out:
            out.writelines(output)

```

## Using GridDB Python Client

Visit https://griddb.net/en/downloads/ to download GridDB Server and follow the installation instructions at the website to
install it. Unfortunately, there is no version of GridDB for Windows OS but you can download Oracle's Virtual Machine to use the
other operating systems on your Windows PC. Follow the instructions at https://www.youtube.com/watch?v=sB_5fqiysi4 to create a Virtual Machine and download the Ubuntu for Windows iso file which is about 2.7GB to use Ubuntu on your PC. 

However, if you'd like to go another way, you can use your Android phone to install the GridDB Server and Client on the phone using a Linux Emulator called **Termux**. This app is like having a Linux OS on your phone and you can install a lot of packages
using the app. The first packages you would want to install are **wget** and **git** as they would allow you to get information
from GitHub. Then run the commands at the terminal to have the GridDB server installed. Also, follow the installation instructions at https://griddb.net/en/blog/data-visualization-with-python-matplotlib-and-griddb/ to install SWIG, GridDB C Client and GridDB Python Client.

Note that you can't install Python's GridDB Client without having SWIG and the C_Client installed. Visit http://prdownloads.sourceforge.net/swig/swigwin-4.0.2.zip to download the version for Windows or http://prdownloads.sourceforge.net/swig/swig-4.0.2.tar.gz to download the version for other operating systems like CentOS and
Ubuntu.

Before you can use the Python Client, you have to run the server in advance and you should get the host, port, cluster name 
and user name noted down because you would use them in your Python Client code. Fill in the code below where you have the 
details mentioned above.

If you're using Jupyter notebook, you may need to follow the instructions at https://griddb.net/en/blog/using-python-to-interface-with-griddb-via-jdbc-with-jaydebeapi/ to get GridDB running on your notebook.

We would use GridDB as a container for our data, fill in the details of your GridDB server where you have empty quotes.


```python
import griddb_python as griddb
factory = griddb.StoreFactory.get_instance()
your_host = ""
your_port = ""
your_cluster_name = ""
your_username = ""
your_password = ""
try:
    gridstore = factory.get_store(host=your_host, port=your_port, 
            cluster_name=your_cluster_name, username=your_username, 
            password=your_password)
conInfo = griddb.ContainerInfo("Phishing Websites",
                    [              
                     ["having_IP_Address", griddb.Type.INTEGER],
                     ["URL_Length", griddb.Type.INTEGER],
                     ["Shortining_Service", griddb.Type.INTEGER],
                     ["having_At_Symbol", griddb.Type.INTEGER],
                     ["double_slash_redirecting", griddb.Type.INTEGER],
                     ["Prefix_Suffix", griddb.Type.INTEGER],
                     ["having_Sub_Domain", griddb.Type.INTEGER],
                     ["SSLfinal_State", griddb.Type.INTEGER],
                     ["Domain_registeration_length", griddb.Type.INTEGER],
                     ["Favicon", griddb.Type.INTEGER],
                     ["port", griddb.Type.INTEGER],
                     ["HTTPS_token", griddb.Type.INTEGER],
                     ["Request_URL", griddb.Type.INTEGER],
                     ["URL_of_Anchor", griddb.Type.INTEGER],
                     ["Links_in_tags", griddb.Type.INTEGER],
                     ["SFH", griddb.Type.INTEGER],
                     ["Submitting_to_email", griddb.Type.INTEGER],
                     ["Abnormal_URL", griddb.Type.INTEGER],
                     ["Redirect", griddb.Type.INTEGER],
                     ["on_mouseover", griddb.Type.INTEGER],
                     ["RightClick", griddb.Type.INTEGER],
                     ["popUpWidnow", griddb.Type.INTEGER],
                     ["Iframe", griddb.Type.INTEGER],
                     ["age_of_domain", griddb.Type.INTEGER],
                     ["DNSRecord", griddb.Type.INTEGER],
                     ["web_traffic", griddb.Type.INTEGER],
                     ["Page_Rank", griddb.Type.INTEGER],
                     ["Google_Index", griddb.Type.INTEGER],
                     ["Links_pointing_to_page", griddb.Type.INTEGER],
                     ["Statistical_report", griddb.Type.INTEGER],
                     ["Result", griddb.Type.INTEGER],                    
                    ],
                    griddb.ContainerType.COLLECTION, True)
    cont = gridstore.put_container(conInfo)   
    data = pd.read_csv("dataset.csv")
    
    #Add data
    for i in range(len(data)):
        ret = cont.put(data.iloc[i, :])
    print("Successfully added the data")
except griddb.GSException as e:
    for i in range(e.get_error_stack_size()):
        print("[", i, "]")
        print(e.get_error_code(i))
        print(e.get_location(i))
        print(e.get_message(i))
```


```python
import pandas as pd
dataset = pd.read_csv("dataset.csv")
```

### Printing the head of the dataset (showing the top five records of the data)

A good thing to understand about the dataset is that all the features are numeric, so there would be no need to encode the data.
Also, the dataset has only three values - **1**, **-1** and **0**; this are just the forms of representing booleans of the data.
If the value is **1**, then the condition (attribute/column name) is true; if the value is **0** or **-1**, it is false.
So, if there is a value of **1** under the **popUpWindow** column, it means that there is a popUpWindow. All these values determine what the result would be, the **Result** column also has **1** and **-1** values which represent **Phishing Website** and 
**Not a Phishing Website** respectively.

Use **print(dataset.head())** if you are using the Python IDLE


```python
""" You can also decide to show the bottom five records of the data by 
typing dataset.tail()"""
dataset.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>having_IP_Address</th>
      <th>URL_Length</th>
      <th>Shortining_Service</th>
      <th>having_At_Symbol</th>
      <th>double_slash_redirecting</th>
      <th>Prefix_Suffix</th>
      <th>having_Sub_Domain</th>
      <th>SSLfinal_State</th>
      <th>Domain_registeration_length</th>
      <th>Favicon</th>
      <th>...</th>
      <th>popUpWidnow</th>
      <th>Iframe</th>
      <th>age_of_domain</th>
      <th>DNSRecord</th>
      <th>web_traffic</th>
      <th>Page_Rank</th>
      <th>Google_Index</th>
      <th>Links_pointing_to_page</th>
      <th>Statistical_report</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>0</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>1</td>
      <td>...</td>
      <td>-1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>0</td>
      <td>-1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 31 columns</p>
</div>



## Building our Model to Predict the Results 

This stage is where we build a model to predict if a website is a phishing website or not
We will use two different classifiers - Logistic Regression and Decision Trees. Though the latter is preferable but building two
different models with these classifiers would make us compare the predictions. The result (predictions) would make us detect
phishing websites whenever we deploy it.
The model building would be in two parts, the first part is where we build a Logistic Regression model while the second part is
where we build a Decision Tree Classifier

## Part 1

### Building a Logistic Regression Classifier model

Now, the dataset has rows and columns. The last column is the result of the predictions if the values in all other columns are
true. In other words, the last column depends on all other columns from the first to the penultimate column and it is
called the **Dependent Feature**. All the other columns are called **Independent Features**.
We are going to define two variables, **X** and **y** to store the values of the independent and dependent features respectively. So, **X** is referred to as the **Independent Variable** while **y** is the (what? Yes) **Dependent Variable**.

The syntax for selecting the values is **data.iloc[number_of_rows_to_select, number_of_columns_to_select].values**. Since we are selecting all the rows, we would slice the dataset values from the first row to the last row using the slice operator **:**, but
for X, we only need all the columns except the last column (result); so we are going to slice to leave out the last column. For y, we need the just the last column so there is no need for slicing.  


```python
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```


```python
print(X)
```

    [[-1  1  1 ...  1  1 -1]
     [ 1  1  1 ...  1  1  1]
     [ 1  0  1 ...  1  0 -1]
     ...
     [ 1 -1  1 ...  1  0  1]
     [-1 -1  1 ...  1  1  1]
     [-1 -1  1 ... -1  1 -1]]
    


```python
print(y)
```

    [-1 -1 -1 ... -1 -1 -1]
    

### Splitting the dataset into Training Set and Test Set


```python
from sklearn.model_selection import train_test_split as tts
```


```python
X_train, X_test, y_train, y_test = tts(X, y, test_size=0.30, random_state=0)
```

### Training a Logistic Regression model on the Training Set


```python
from sklearn.linear_model import LogisticRegression
```


```python
classifier = LogisticRegression(random_state=0, solver='lbfgs')
```


```python
classifier.fit(X_train, y_train)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='warn', n_jobs=None, penalty='l2',
                       random_state=0, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)




```python
y_pred = classifier.predict(X_test)
```


```python
print(y_pred)
```

    [-1 -1  1 ... -1  1  1]
    


```python
# Accuracy of the model
acc = classifier.score(X_test, y_test)
acc *= 100
accu = round(acc, 2)
accuracy = str(accu)+"%"
print(
f"""
The Accuracy of our model, using the Logistic Regression Classifier
{accuracy}
"""
)
```

    
    The Accuracy of our model, using the Logistic Regression Classifier
    92.28%
    
    

## Making the Confusion Matrix


```python
from sklearn.metrics import confusion_matrix
cMatrix = confusion_matrix(y_test, y_pred)
print(cMatrix)
```

    [[1341  157]
     [  99 1720]]
    

## Visualizing Data

GridDB also provides powerful Data Visualization, follow the instructions at https://griddb.net/en/blog/data-visualization-with-python-matplotlib-and-griddb/ to visualiza the data using GridDB and Python's matplotlib. Seaborn is used here and this takes
a great deal of time to visualiza because of the numerous attributes the dataset has.


```python
import seaborn as sns
sns.pairplot(dataset)
```




    <seaborn.axisgrid.PairGrid at 0xa639f10>




![png](output_45_1.png)


From the visualization above, we would see that the **age_of_domain** column correlates with the Result column.

## Part 2

### Building a Decision Tree Classifier Model


```python
from sklearn.tree import DecisionTreeClassifier
```


```python
dtClassifier = DecisionTreeClassifier()
```


```python
dtClassifier.fit(X_train, y_train)
```




    DecisionTreeClassifier(class_weight=None, criterion='gini', max_depth=None,
                           max_features=None, max_leaf_nodes=None,
                           min_impurity_decrease=0.0, min_impurity_split=None,
                           min_samples_leaf=1, min_samples_split=2,
                           min_weight_fraction_leaf=0.0, presort=False,
                           random_state=None, splitter='best')




```python
dty_pred = dtClassifier.predict(X_test)
```


```python
print(dty_pred)
```

    [-1 -1 -1 ... -1  1  1]
    


```python
from sklearn.metrics import confusion_matrix
cMatrix = confusion_matrix(y_test, dty_pred)
print(cMatrix)
```

    [[1425   73]
     [  51 1768]]
    


```python
# Accuracy of the model
dtacc = dtClassifier.score(X_test, y_test)
dtacc *= 100
dtaccu = round(dtacc, 2)
dtaccuracy = str(dtaccu)+"%"
print(
f"""
The Accuracy of our model, using the Decision Tree Classifier
{dtaccuracy}
"""
)
```

    
    The Accuracy of our model, using the Decision Tree Classifier
    96.26%
    
    

Now, it is a very good practice to use GridDB for Machine Learning. It allows us to store the dataset in a container and also has the capability of storing Machine Learning models. So, in case you want to use your model in a project, GridDB would be a good choice since you can also access the model you stored. More information about GridDB is available at https://www.griddb.net/
