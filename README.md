EXPLAINING AI MODELS WITH DEEP ENSEMBLE LEARNING TECHNIQUES

MOTIVATION:
The major disadvantage of the existing deep learning techniques is the trade off 
of the models between explainability and performance. This means that the 
algorithms are becoming increasingly complex to be understood in the process 
of decision making. The motivation of the research is the development of deep 
ensemble learning techniques using Explainable AI (XAI). The AI models are to 
be examined with regard to their explanability. The aim is to develop key figures 
for explainability of the respective AI model and to develop strategies to 
improve the explainability of these models.
1. Research and understand the existing deep ensemble models
2. Use a test dataset to explain the deep ensemble learners with the help of 
two approaches i.e. LIME and SHAP
3. Evaluate the explanations locally with the help of a framework called LEAF
Initially, the test dataset is fitted to the different deep ensemble learning models 
by evaluating the accuracy of the models. Accuracy of the model plays an 
important role to move to the next step. The next step would be to explain the 
models with LIME and SHAP approaches. The explainers are evaluated with the 
aid of LEAF, which employs a variety of metrics to determine whether the 
explanation is locally accurate

DATASET INFORMATION:
The dataset is an automotive dataset where the 7 features are the 
important parameters of the Hybrid Electric Vehicle (HEV) and the 5 
labels indicating the 5 classes are the faults that are responsible for the 
failure of the HEV, that need to be classified by the ensemble models. The 
dataset can be described as follows:
1 CSV file: 7 columns, 151 rows per simulation
Sample time: 0.1s
Number of samples per feature per simulation: 151 (15 seconds starting 
at 1 second before fault injection time)
Number of features: 7 (‘Motor_Current’, ‘Generator_Current’, 
‘Engine_Torque’, ‘Vehicle_Speed’, ‘Generator_Speed’, ‘Battery_Current’, 
‘Engine_Speed’)
Number of labels: 5 ('0 No fault','1 Motor fault','2 Generator fault','3 
Battery fault','4 ICE fault')
Number of datasets (CSV files) per label: 400 (some labels have less sets, 
e.g., if a component is not used in a certain situation and is turned off by 
the mode logic, it cannot be diagnosed and the data would be misleading 
for the learning algorithm)

For Exploratory Data Analysis (EDA), the seed is set for randomly loading 
the dataset. The number of classes and the data directory is defined for 
easy handling of the input data by the algorithm. The resulting classes 
(labels) of the dataset are defined and the data is loaded by using the label 
number. The empty cells and the duplicated rows are taken care of in this 
Exploratory Data Analysis step. The label number is used to create the 
label data i.e the resulting class that needs to be classified. A new 
dataframe is created according to the label numbers and train and test 
split operations is used to create the training and testing data both for 
input features and the output classes. EDA plays an important role in any 
machine learning problem, as mishandling of data might severely affect 
the accuracy of the model as well as the predictions and the classification results

RESULTS:
Deep Neural Decision Tree Train acc- 72% Test acc- 74%.
Deep Neural Decision Forest Train acc- 72% Test acc- 76%.
Deep Forest Train acc- 74% Test acc- 77%.
Deep SVM Train acc- 80% Test acc- 82%.
Explainers LIME & SHAP compatible with all 4 ensemble learners.
But the framework used for comparison LEAF is not compatible with DNDT and DNDF.
Hence, suggested a tenative design of a new framework and also some metrics that can be improved in the current framework.
The results are obtained via binary comaprison method for e.g. 0th vs 1st class, 0th vs 2nd class, 0th vs 3rd class and 0th vs 4th class.
In this case, the results (PNG image files) uploaded here are only for the first case i.e. 0th (No fault) vs 1st (Motor fault).
DF=Deep Forest, DNDT=Deep Neural Decision Tree, DNDF=Deep Neural Decision Forest, DSVM=Deep Support Vector Machine


RUNNING THE CODE:
Run the main.py file which will run the code//
Compare.py checks the respective ensemble learners with LIME and SHAP//
Get_Data.py gets the data and performs data wrangling//
leaf.py is the leaf framework from scratch//
Remaining .py files are the ensemble learners written individually and called via main.py//

