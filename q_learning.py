import numpy as np
import sys
import pandas as pd

class td_qlearning:

    def __init__(self, trajectory_filepath):
        # trajectory_filepath is the path to a file containing a trajectory through state space
        # Return

        self.alpha = 0.1
        self.gamma = 0.5

        #dictionary to store all state-action pairs with their q values
        self.q_dict = {}

        #read csv file into pandas dataframe
        df = np.loadtxt(trajectory_filepath, delimiter=",", dtype=str)
        print(df)
        #for loop to access state/action pair from each row in dataframe
        #this for loop will also calculate the q value for each pair and update the dictionary created above
        for i in range(len(df)-1) :
            #assigning value to state and action by accessing individual elements from dataframe
            state = df[i][0]
            action = df[i][1]

            #repeat steps above for next state/action pair in trajectory
            state_nxt = df[i+1][0]
            action_nxt = df[i+1][0]

            #get the number of dirty squares
            dirty_squares = 0
            for element in range(1,6):
                if str(state)[element] == "1":
                    dirty_squares += 1

            #multiply number of dirty sqaures to -1 in order to calculate reward
            Rs = -1 * dirty_squares

            #find max value for state_nxt from all possible actions
            max_nstate_a = 0

            if int(str(state_nxt)[0]) == 1 :
                max_nstate_a = max(self.qvalue(state_nxt, "D"), self.qvalue(state_nxt, "C"))
            elif int(str(state_nxt)[0]) == 2 :
                max_nstate_a = max(self.qvalue(state_nxt, "R"), self.qvalue(state_nxt, "C"))
            elif int(str(state_nxt)[0]) == 3 :
                max_nstate_a = max(self.qvalue(state_nxt, "D"), self.qvalue(state_nxt, "U"), self.qvalue(state_nxt, "L"), self.qvalue(state_nxt, "R"), self.qvalue(state_nxt, "C"))
            elif int(str(state_nxt)[0]) == 4 :
                max_nstate_a = max(self.qvalue(state_nxt, "L"), self.qvalue(state_nxt, "C"))
            else :
                max_nstate_a = max(self.qvalue(state_nxt, "U"), self.qvalue(state_nxt, "C"))

            #calculate expected long term reward
            expected_result = Rs + self.gamma*max_nstate_a - self.qvalue(state, action)

            #calcualte value for q
            q = self.qvalue(state, action) + self.alpha*expected_result

            #add to/update q value dictionary
            self.q_dict[(state, action)] = q

        #print(self.q_dict)


    def qvalue(self, state, action):
        # state is a string representation of a state
        # action is a string representation of an action

        #set initial estimate of q to 0
        q = 0

        #if an updated value of q exists, return that. If not return the initial estimate
        if (state, action) in self.q_dict:
            q = self.q_dict[(state, action)]

        return q

    def policy(self, state):
        # state is a string representation of a state

        # Return the optimal action under the learned policy
        a = ""
        actions = ["L" , "C" , "R" , "U" , "D"]
        key_list = list(self.q_dict)
        q_v = []
        actions_of_q_v = []

        # Saves the action and it's appropriate q value
        for index, key in enumerate(self.q_dict):
            if key[0] == state:
                q_v.append(self.q_dict.get(key))
                actions_of_q_v.append(key_list[index][1])

        # If the state is unique to the trajectory return the action of the next state
        if len(q_v) == 1:
            for act in actions:
                try:
                    temp_state_pair = (state, act)
                    temp_index = key_list.index(temp_state_pair)
                    a =  key_list[temp_index+1][1]
                    return a
                except:
                    continue

        # Otherwise, return the action of the state with the highest q value
        temp = q_v.index(max(q_v))
        a = actions_of_q_v[temp]
        return a
