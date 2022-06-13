# C7081 - Predicting the outcome of a League of Legends game from the first 10 minutes of play
This is the repository for the C7081 Statistical Learning Assignment


# Background
League of legends is a team-based game developed by the company Riot Games for PC. The game is a red
verses blue objective based game where player pick characters with differing abilities to attempt to gain an
advantage. The objective of the game is to destroy defending towers and kill enemy minions to gain entry
to the enemy team base and destroy the “Nexus”. Killing minions and objectives, such as towers and other
elite monsters, rewards players with gold which is used to buy items which enhance the characters attack,
defence, hit points and ability powers. The game has a following of around 115 million competitive players
and splits these up into ranks from bronze to challenger (Kou and Gui, 2020), ranks are shown in table 1
below.

*Table 1: Player base and Rank*

| Rank        | Player percentage (%) |
|-------------|-----------------------|
| Iron        | 7.1                   |
| Bronze      | 22.0                  |
| Silver      | 35.0                  |
| Gold        | 23.0                  |
| Platinum    | 7.9                   |
| Diamond     | 2.5                   |
| Master      | 0.032                 |
| Grandmaster | 0.040                 |
| Challenger  | 0.017                 |

It can be seen from the table that the top three ranks make up less than 1% of the total playing population
of the game, these tend to be sporting professional who compete globally.

The game can be broken up into three segments: Early game, Mid game, and late game, where the prediction
of who will in the game can swing from team to team. Early game is defined as the first 20 minutes where, for
the first 15 minutes, there is an inability to surrender (forfeit) the game; a unanimous decision to surrender
must be achieved between the 15-to-20-minute mark. For this report the data for the first 10 minutes of
games is to be used for games in the platinum ranking. With the outcome of the game being successfully
predicted after these 10 minutes, clarity can be given to players who are unsure whether forfeiting and
starting a new game is the correct decision.

It has been shown that video games and esports share a link to the human desire to gamble (Macey and
Hamari, 2018; Fisher and Griffiths, 1995; Johansson and Gotestam, 2004; Wood et al., 2004). Understanding
the risks and wanting to swing the odds in favour of the player is well documented (Ore, 2017), and first
published in Cardano’s Book on Games of Chance in 1564 (Cardano, 2015). Incorporating the use of
probability and understanding when to forfeit can save a player money, but more commonly time.

This report will endeavour to successfully predict the outcome of League of Legends game played in the
platinum rank dependant on the outcomes of variables in the first 20 minutes of a game, giving the player an
understanding of the odds comparable to what Cardano achieved in 1564. A similar study, conducted by de
Souza and Cortimiglia (2017), was seen to achieve a 75% accuracy in predicting the outcome of a league of
legends game using Logistic Regression and Random Forests; 75% accuracy will be taken as the benchmark
for success in the methods implemented for this report.

## Objectives
* Find the highest accuracy possible from a range of models to see if we can successfully predict the
outcome of the game looking only at variables from the first 10 minutes.
* State the most influential variables to the outcome of the game and theorise as to why this might be.

# Methods
## Data
The data was taken from the Kaggle dataset search engine and contains 9880 data points. A subset of 30
variable were taken forward from the original data set as some were removed as they were seen to have little
relevance on the outcome of the game from a user’s perspective. Sub setting occurred when the data was
turned into tidy form in Microsoft Excel. The data contains data points recorded for the first 10 minutes of
games in the diamond rank of Leagues of Legends for both the blue and red team. The variables and their
description can be found in table 2.

*Table 2: Table of variables and description*

| variable   | Class     | definition                                                               |
|------------|-----------|--------------------------------------------------------------------------|
| game_id    | Factor    | game identification                                                      |
| b_wins     | Numerical | blue team wins, 1 = win, 0 = loss                                        |
| b_war_pl   | Numerical | blue team number of wards placed                                         |
| b_war_des  | Numerical | blue team number of wards destroyed                                      |
| b_fir_blo  | Factor    | blue team first kill of the game, 1 = yes, 0 = no                        |
| b_k        | Numerical | blue team total kills                                                    |
| b_d        | Numerical | blue team total deaths                                                   |
| b_a        | Numerical | blue team total assists                                                  |
| b_e_mon    | Numerical | blue team elite monsters killed                                          |
| b_tow_des  | Numerical | blue team towers destroyed                                               |
| b_tot_gp   | Numerical | blue team total gold                                                     |
| b_av_lvl   | Numerical | blue team average level                                                  |
| b_tot_xp   | Numerical | blue team total experience points                                        |
| b_tot_mons | Numerical | blue team total minions killed                                           |
| b_cs_min   | Numerical | blue team total minions, monsters and wards destroyed per minute of play |
| b_gp_min   | Numerical | blue team gold earned per minute of play                                 |
| r_war_pl   | Numerical | red team number of wards placed                                          |
| r_war_des  | Numerical | red team number of wards destroyed                                       |
| r_fir_blo  | Factor    | red team first kill of the game, 1 = yes, 0 = no                         |
| r_k        | Numerical | red team total kills                                                     |
| r_d        | Numerical | red team total deaths                                                    |
| r_a        | Numerical | red team total assists                                                   |
| r_e_mon    | Numerical | red team elite monsters killed                                           |
| r_tow_des  | Numerical | red team towers destroyed                                                |
| r_tot_gp   | Numerical | red team total gold                                                      |
| r_av_lvl   | Numerical | red team average level                                                   |
| r_tot_xp   | Numerical | red team total experience points                                         |
| r_tot_mons | Numerical | red team total minions killed                                            |
| r_cs_min   | Numerical | red team total minions, monsters and wards destroyed per minute of play  |
| r_gp_min   | Numerical | red team gold earned per minute of play                                  |

The data was read into RStudio (2021) using the readxl package due to the tidy data being created in
Microsoft Excel. The data frame was initially put into a pairs plot to visualise correlation; however, the
number of variables and data points made this challenging. An approach was taken to look at half the
variables at a time against the dependent variable of Blue team wins. This saved computing power and
made the pairwise plot more readable. Within this plot, the outcome of the game was highlighted using
colour for the Blue team wins variable. This was done for both the blue and red team variables.

Boxplots were made to view independent variables individually against the dependent variable. This led into
the analysis of the data where the summary of the data frame was printed.

Correlation was checked to test for collinearity, as there was evidence of this it was decided that some
variables should be removed. In a study by Dormann et al. (2012) it was shown that collinearity and cause
severe problems when a model is trained on data with non-independent data, as a test for validity a model
was run without collinear variables removed, a total of 8 variables were removed.


## Binary Logistical Regression
It was decided that due to the dependent variable being a win or lose outcome that a binomial logistic
regression would be appropriate for the data set. To correctly use the logistical regression, it was necessary
to use the logistic function, this prevented values for the probability from being negative or greater than one,
this also meant that the method of maximum likelihood was used due to its statistical properties (James et
al.,2013).

The data was then split into test and train data using an 80:20 split of the data. This put 7902 data points
into the train data and 1977 in the test.

The model showed that there were six significant predictor variables table 2. As two of the variables with
the lowest p-values were total gold earned by each team it can be noticed that this might have the largest
influence on the outcome of the game.

*Table 3: Significant variables*

| Variable | P-Value   |
|----------|-----------|
| b_e_mon  | 3.10e-06  |
| b_tot_gp | 2.70e-11  |
| b_tot_xp | 8.97e-05  |
| b_cs_min | 0.040613  |
| r_a      | 0.049410  |
| r_e_mon  | 2.20e-06  |
| r_tot_gp | 7.33e-11  |
| r_tot_xp | 0.000341  |

Validation of the binomial logistical model was completed where the variance of the residuals was plotted to
check that 95% of the data points lay within +/- 2 standard error of the mean.

The standard residual plots were also viewed, as seen in Figure 2, and it was found that the data was normally
distributed with few outliers.

![](https://github.com/BUCKERS99/C7081-Assignment/blob/main/Plots/stan_resid.png)

*Figure 2: Standard residual plots*

## Stepwise Regression
Stepwise regression was chosen for exploratory analysis as the backwards selection of variables would allow
for the variables with the least effect will be removed. Stepwise regression does have its issues; it has been
reported that the forward method of stepwise regression can produce suppressor effects within variables. It
also can cause issues as it underestimates the relationships between variables when it removes them from the
model (McElreath, 2018). Results from the stepwise model should be validated carefully and both forward
and backwards stepwise were used to determine the difference in accuracy. This method was chosen due to
its wide use in determining accuracy in models.

Validation error depending on model size was viewed here for the types of regression used, this is shown in Figure 3.

![](https://github.com/BUCKERS99/C7081-Assignment/blob/main/Plots/val_err.png)

*Figure 3: Validation error of stepwise regression*

## Tree and Random Forest
Both tree and random forest methods were implemented in this study to determine their accuracy on the
outcome. As the dependant variable was binary, a classification regression method was used. In this it was
necessary to work out a value for m, as this is the number of predictors used at each split. For classification
problems, like the one we are studying, it is considered correct to take m = square root of p when p = the
total number of predictors. In this study m = 4. As random forest was a method used to attain a 75%
accuracy in the study conducted by de Souza and Cortimiglia (2017) it was used to assess its outcome using
this dataset.

# Results
After running the binary logistic regression it is seen that there are 10 variables that show significance in the
result of the blue team winning or losing. The most significant variables, shown by the lowest p-value, are the
total gold for both blue and red team. Accuracy of the model was attained by showing test data to the model
and validating the outcome with the original data set. K – nearest neighbour was seen to produce the highest
accuracy with 72.25% closely followed by forward stepwise regression with 72.23%, as shown in Figure 4.

![](https://github.com/BUCKERS99/C7081-Assignment/blob/main/Plots/model_accuracy_plot.png)

*Figure 4: Model accuracy plot*

Interestingly the decision tree and random forest methods used did not achieve a high accuracy, with the
study of de Souza and Cortimiglia (2017) producing their results using this method it would be worthwhile
comparing the methods and datasets used. It should be noted that the data used in this study only records
within the first 10 minutes of the game. Games can last for over 60 minutes and this is where the 2017 study
could have different results.

The results of the backwards stepwise model show us that the variable total gold for both teams has the
lowest p value, this is consistent throughout all the models; for example, in the decision tree method it was
the only variable that was chosen to make the tree. Taking this into account, it can be said with confidence
that total gold earned by a team in the first 10 minutes of a game highly influences the outcome of the game.
Gold allows players to buy items which improves damage as well as other major character abilities giving
the player with the most items purchased a distinct advantage over others.

It should be taken into consideration that a lot can happen in the rest of the game, and the outcome can
be influenced by other factors: such as players, deliberately or unintentionally, leaving the game; players
making mistakes; or players having picked a wrong match up between characters.

# Conclusion
From the range of models used it was seen that the k-nearest neighbour model produced the highest accuracy
regarding outcome of the game. Stepwise regression is a method that is used commonly when reporting
accuracy of outcomes which is used by current governments to inform policy, but there are issues regarding
its use. As a 75% accuracy was stated as the benchmark in reporting the accuracy it can be said that this
study did not hit the benchmark. Were this study to be improved a Bayesian method might be used to
predict the outcome. It would also be useful to compare the results from other ranks of league of legends to
see if the variables that influence the outcome are similar.

# References
Kou, Y. and Gui, X., 2020. Emotion Regulation in eSports Gaming: A Qualitative Study of League of
Legends. Proceedings of the ACM on Human-Computer Interaction, 4(CSCW2), pp.1-25.

Souza, R.T.D., and Cortimiglia, M.N. 2017. Aplicacao de algoritmos classificadores para previsao de vitoria
em uma partida de League of Legends.

Ore, O., 2017. Cardano: The gambling scholar (Vol. 5063). Princeton University Press.

Cardano, G., 2015. The book on games of Chance: the 16th-century treatise on probability. Courier Dover
Publications.

Johansson, A. and Gotestam, K.G., 2004. Problems with computer games without monetary reward: similarity to pathological gambling. Psychological reports, 95(2), pp.641-650.

Wood, R.T., Gupta, R., Derevensky, J.L. and Griffiths, M., 2004. Video game playing and gambling in
adolescents: Common risk factors. Journal of Child & Adolescent Substance Abuse, 14(1), pp.77-100.

RStudio Team (2021). RStudio: Integrated Development for R. RStudio, PBC, Boston, MA URL.

Dormann, C.F., Elith, J., Bacher, S., Buchmann, C., Carl, G., Carre, G., Marquez, J.R.G., Gruber, B.,
Lafourcade, B., Leitao, P.J. and Munkemuller, T. 2013. Collinearity: a review of methods to deal with it
and a simulation study evaluating their performance. Ecography, 36(1), pp.27-46.

James, G., Witten, D., Hastie, T. and Tibshirani, R., 2013. An introduction to statistical learning (Vol.
112, p. 18). New York: springer.

McElreath, R., 2018. Statistical rethinking: A Bayesian course with examples in R and Stan. Chapman and
Hall/CRC.
