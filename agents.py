from cmath import inf
import random
import math
import connect383

BOT_NAME =  "ConnectGO"


class RandomAgent:
    """Agent that picks a random available move.  You should be able to beat it."""
  
    rseed = None  # change this to a value if you want consistent random choices

    def __init__(self):
        if self.rseed is None:
            self.rstate = None
        else:
            random.seed(self.rseed)
            self.rstate = random.getstate()

    def get_move(self, state):
        if self.rstate is not None:
            random.setstate(self.rstate)
        return random.choice(state.successors())


class HumanAgent:
    """Prompts user to supply a valid move.  Very slow and not always smart."""

    def get_move(self, state, depth=None):
        move__state = dict(state.successors())
        prompt = "Kindly enter your move {}: ".format(sorted(move__state.keys()))
        move = None
        while move not in move__state:
            try:
                move = int(input(prompt))
            except ValueError:
                continue
        return move, move__state[move]


class MinimaxAgent:
    """Artificially intelligent agent that uses minimax to optimally select the best move."""

    def get_move(self, state):
        """Select the best available move, based on minimax value."""
        nextp = state.next_player()
        best_util = -math.inf if nextp == 1 else math.inf
        best_move = None
        best_state = None

        for move, state in state.successors():
            util = self.minimax(state)
            if ((nextp == 1) and (util > best_util)) or ((nextp == -1) and (util < best_util)):
                best_util, best_move, best_state = util, move, state
        return best_move, best_state

    def minimax(self, state):
        """Determine the minimax utility value of the given state.

        Gets called by get_move() to determine the value of each successor state.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the exact minimax utility value of the state
        """
        if(state.is_full()):
            return state.utility()
        children =  state.successors()
        minChild = inf
        maxChild = -inf
        for child in children:
            childUtility = self.minimax(child[1])
            if(childUtility>maxChild):
                maxChild = childUtility
            if(childUtility<minChild):
                minChild = childUtility
        if(state.next_player() == 1):
            return maxChild
        return minChild


class MinimaxLookaheadAgent(MinimaxAgent):
    """Artificially intelligent agent that uses depth-limited minimax to select the best move.
 
    Hint: Consider what you did for MinimaxAgent. What do you need to change to get what you want? 
    """

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state.

        Gets called by get_move() to determine the value of successor states.

        The depth data member (set in the constructor) determines the maximum depth of the game 
        tree that gets explored before estimating the state utilities using the evaluation() 
        function.  If depth is 0, no traversal is performed, and minimax returns the results of 
        a call to evaluation().  If depth is None, the entire game tree is traversed.

        Args:
            state: a connect383.GameState object representing the current board

        Returns: the (possibly estimated) minimax utility value of the state
        """
        if(self.depth_limit == 0):
            return self.evaluation(state)
        currDepth = state.num_cols * state.num_rows
        num_obstacles = 0
        for row in state.get_rows():
            for x in row:
                if(x==0):
                    currDepth -= 1
        if(self.depth_limit + currDepth >= state.num_cols * state.num_rows):
            lateral = MinimaxAgent()
            return lateral.minimax(state)
        return self.minimax_depth(state,0)
        

    def minimax_depth(self, state, depth):
        if(depth == self.depth_limit):
            return self.evaluation(state)
        children =  state.successors()
        minChild = inf
        maxChild = -inf
        for child in children:
            childUtility = self.minimax_depth(child[1],depth+1)
            if(childUtility>maxChild):
                maxChild = childUtility
            if(childUtility<minChild):
                minChild = childUtility
        if(state.next_player() == 1):
            return maxChild
        return minChild       

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        Gets called by minimax() once the depth limit has been reached.  
        N.B.: This method must run in "constant" time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        """avgChildUtility = 0
        children = state.successors()
        temp = 0
        for child in children:
            avgChildUtility += child[1].utility()
            temp += 1
        avgChildUtility /= temp"""
        potentialPoints = 0
        lastStreak = None
        for column in state.get_cols():
            for streak in connect383.streaks(column):
                lastStreak = streak
            if column.count(0)>0:
                potentialPoints += lastStreak[0]*(lastStreak[1]**2)
        for row in state.get_rows():
            for streak in connect383.streaks(row):
                lastStreak = streak
            if row.count(0)>0:
                potentialPoints += lastStreak[0]*(lastStreak[1]**2)
        for diag in state.get_diags():
            for streak in connect383.streaks(diag):
                lastStreak = streak
            if diag.count(0)>0:
                potentialPoints += lastStreak[0]*(lastStreak[1]**2)
        return state.utility()*.6+potentialPoints*.4


class AltMinimaxLookaheadAgent(MinimaxAgent):
    """Alternative heursitic agent used for testing."""

    def __init__(self, depth_limit):
        self.depth_limit = depth_limit

    def minimax(self, state):
        """Determine the heuristically estimated minimax utility value of the given state."""
        if(self.depth_limit == 0):
            return self.evaluation(state)
        currDepth = state.num_cols * state.num_rows
        num_obstacles = 0
        for row in state.get_rows():
            for x in row:
                if(x==0):
                    currDepth -= 1
        if(self.depth_limit + currDepth >= state.num_cols * state.num_rows):
            lateral = MinimaxAgent()
            return lateral.minimax(state)
        return self.minimax_depth(state,0)
        

    def minimax_depth(self, state, depth):
        if(depth == self.depth_limit):
            return self.evaluation(state)
        children =  state.successors()
        minChild = inf
        maxChild = -inf
        for child in children:
            childUtility = self.minimax_depth(child[1],depth+1)
            if(childUtility>maxChild):
                maxChild = childUtility
            if(childUtility<minChild):
                minChild = childUtility
        if(state.next_player() == 1):
            return maxChild
        return minChild       

    def evaluation(self, state):
        """Estimate the utility value of the game state based on features.

        Gets called by minimax() once the depth limit has been reached.  
        N.B.: This method must run in "constant" time for all states!

        Args:
            state: a connect383.GameState object representing the current board

        Returns: a heuristic estimate of the utility value of the state
        """
        """avgChildUtility = 0
        children = state.successors()
        temp = 0
        for child in children:
            avgChildUtility += child[1].utility()
            temp += 1
        avgChildUtility /= temp
        return avgChildUtility"""
        return state.utility()


class MinimaxPruneAgent(MinimaxAgent):
    """Computer agent that uses minimax with alpha-beta pruning to select the best move.
    
    Hint: Consider what you did for MinimaxAgent.  What do you need to change to prune a
    branch of the state space? 
    """
    def minimax(self, state):
        """Determine the minimax utility value the given state using alpha-beta pruning.

        The value should be equal to the one determined by MinimaxAgent.minimax(), but the 
        algorithm should do less work.  You can check this by inspecting the value of the class 
        variable GameState.state_count, which keeps track of how many GameState objects have been 
        created over time.  This agent does not have a depth limit.

        N.B.: When exploring the game tree and expanding nodes, you must consider the child nodes
        in the order that they are returned by GameState.successors().  That is, you cannot prune
        the state reached by moving to column 4 before you've explored the state reached by a move
        to column 1 (we're trading optimality for gradeability here).

        Args: 
            state: a connect383.GameState object representing the current board

        Returns: the minimax utility value of the state
        """
        return self.alphabeta(state,-inf,inf)

    def alphabeta(self, state,alpha, beta):
        if(state.is_full()):
            return state.utility()
        maxChild = -inf
        minChild = inf
        if(state.next_player() == 1):
            for child in state.successors():
                childUtility = self.alphabeta(child[1], alpha, beta)
                if(childUtility>maxChild):
                    maxChild = childUtility
                if(childUtility>alpha):
                    alpha = childUtility
                if(beta<=alpha):
                    break
            return maxChild
        else:
            for child in state.successors():
                childUtility = self.alphabeta(child[1], alpha, beta)
                if(childUtility<minChild):
                    minChild = childUtility
                if(childUtility<beta):
                    beta = childUtility
                if(beta<=alpha):
                    break
            return minChild           



def get_agent(tag):
    if tag == 'random':
        return RandomAgent()
    elif tag == 'human':
        return HumanAgent()
    elif tag == 'mini':
        return MinimaxAgent()
    elif tag == 'prune':
        return MinimaxPruneAgent()
    elif tag.startswith('look'):
        depth = int(tag[4:])
        return MinimaxLookaheadAgent(depth)
    elif tag.startswith('alt'):
        depth = int(tag[3:])
        return AltMinimaxLookaheadAgent(depth)
    else:
        raise ValueError("bad agent tag: '{}'".format(tag))       
