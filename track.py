import numpy as np
import gym

class Track():
    def __init__(self, levelSeed = 1, actionRepeat = 8, stackSize = 4, env = 'CarRacing-v0'):
        """ Runs the provided actions on the track
        
            Parameters
            -------
            levelSeed : int, 1
                The seed to create the level with. Change to create different tracks the AI hasn't trained on.
            actionRepeat : int, 8
                The number of times to repeat an action
            stackSize : int, 4
                The frames per state
            env : string, "CarRacing-v0"
                The environment to run
        """
        self.actionRepeat = actionRepeat
        self.stackSize = stackSize
        self.track = gym.make(env)
        self.track.seed(levelSeed)

    def reset(self):
        """ Resets the track to it's original values """
        self.crashed = False
        self.stateCounter = 0
        self.avgRun = self.memoryReward()

        fullColourFrame = self.track.reset()
        greyScaled = self.rgbAsGrayscale(fullColourFrame)
        self.stack = [greyScaled] * self.stackSize
        return np.array(self.stack)

    def step(self, action):
        """ Runs the provided actions on the track
        
            Parameters
            -------
            action : np.ndarray
                A list of actions to perform on the track
            Returns
            -------
                np.array
                    Array with the current framestack, score, finished status and crashed status.
            Raises
            -------
                AssertionError
                    Raised when stacksize is not equal to the length of self.stack
        """

        score = 0
        for i in range(self.actionRepeat):
            fullColourFrame, reward, crashed, _ = self.track.step(action)

            " Don't give it a nudge for dying "
            if crashed:
                reward += 100

            " Give it a nudge if it's driving over grass "
            if np.mean(fullColourFrame[:, :, 1]) > 185.0:
                reward -= 0.05
            score += reward

            " Check if it's getting rewards, if not end and retry "
            finished = True if self.avgRun(reward) <= -0.1 else False
            if finished or crashed:
                break

        greyScaled = self.rgbAsGrayscale(fullColourFrame)
        self.stack.pop(0)
        self.stack.append(greyScaled)
        assert self.stackSize == len(self.stack), "Stacksize is not equal to the length of the stack!"
        return np.array(self.stack), score, finished, crashed

    def render(self):
        """  Function that renders the OpenAI environment """
        self.track.render()

    @staticmethod
    def memoryReward():
        """ Calculates the mean reward for keeping previous runs in memory
        
            Returns
            -------
            np.ndarray
                Mean memory reward over all saved previous states.
        """
        stateCount = 0
        memSize = 100
        previousStates = np.zeros(memSize)

        def memory(reward):
            """ The memory function calculates the mean for the previous states as a memory reward
                Parameters
                -------
                reward : float
                    The reward per frame.
                Returns
                -------
                np.ndarray
                    Mean memory reward over all saved previous states.
            """

            nonlocal stateCount
            previousStates[stateCount] = reward
            stateCount = (stateCount + 1) % memSize
            return np.mean(previousStates)

        return memory

    @staticmethod
    def rgbAsGrayscale(rgb, colorNormalization=True):
        """ Converts an rgb frame to a greyscale frame
            
            Parameters
            -------
            rgb : np.ndarray
                The RGB frame as an ndarray
            colorNormalization : boolean, True
                Enable color nomalization.
            Returns
            -------
            grey : np.ndarray
                A vector with all of it's RGB values turned to greyscale.
        """
        grey = np.dot(rgb[..., :], [0.299, 0.587, 0.114])
        if colorNormalization:
            grey = grey / 128. - 1.
        return grey