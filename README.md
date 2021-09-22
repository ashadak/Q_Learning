# Q_Learning
The project demonstrates Bayes theorem and rule-based systems.Consider a vacuum cleaner agent working in a room with five squares, as illustrated below. In this project, we
will use temporal difference Q-learning to learn the optimal policy for the vacuum cleaner.

![image](https://user-images.githubusercontent.com/59786588/134397706-da1d53e9-e3b7-4455-be15-187ef68843ff.png)

Assume that the vacuum cleaner starts in the Square 1. It can take the following actions: clean the current square,
move to a horizontally or vertically adjacent square.

Each square can either have the state dirty or clean (this is a binary state; there is no degree of dirtiness).

Assume that this is a fully observable environment. That is, the vacuum cleaner knows for all times its position and
whether each square is clean or dirty.

Assume that this is a discrete time environment. At each time, the agent may take one action. Furthermore, at
each time, with some probability, one square’s state may change from clean to dirty.

The following string representations will be used to specify states of the squares:
PS1S2S3S4S5
Where P is the square number the vacuum cleaner is in
Si = 0 if the Square i is clean; 1 if the Square i is dirty

The following string representations will be used to specify actions of the vacuum cleaner:
“C”: clean current square
“L”: move left
“R”: move right
“U”: move up
“D”: move down

The reward r associated with a state s is:
r(s) = -1 * number of dirty squares

The following parameters will be used:
Gamma = 0.5 (discount factor)
Alpha = 0.1 (learning rate)

Initially the Q-function will be estimates as:
Q(s, a) = 0


The “td_qlearning” class has three member functions: “__init__”, “qvalue”, and “policy”.

The function “__init__” is a constructor that takes one input argument. The input argument is the full path to
a CSV file containing a single trajectory through the state space. The CSV file will contain two columns, the first with
a string representation of the sate and the second with a string representation of the action taken in that state. The
i'th row of the CSV file indicates the state-action pair at time i.

The function “qvalue” takes two input arguments. The first input argument is a string representation of a state.
The second input argument is the string representation of an action. The function outputs the Q-value
associated with that state-action pair (according to the Q-function learned from the trajectory in the file passed to
the __init__ function).

The function “policy” takes one input argument. The input argument is a string representation of a state. The
function outputs the optimal action (according to the Q-function learned from the trajectory in the file passed
to the __init__ function).

An example of a trajecory file is included in the repo.

More info on Q Learning: https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56 
