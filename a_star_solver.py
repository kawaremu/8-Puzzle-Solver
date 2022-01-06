import pygame,sys
import numpy as np
import time



#Constants 
BOARDWIDTH = 3 # number of columns in the board
BOARDHEIGHT = 3 # number of rows in the board

TILESIZE = 150
WINDOWWIDTH = 800
WINDOWHEIGHT = 600
FPS = 30
BLANK = None


#                  R    G    B
BLACK =          (  0,   0,   0)
WHITE =          (255, 255, 255)
BRIGHTBLUE =     (  0,  50, 255)
DARKTURQUOISE =  (  3,  54,  73)
GREEN =          (  0, 204,   0)
BEAUTIFUL_BLUE = (187, 212, 252)
TILE_COLOR =     (   8, 91, 110)

# This sets the margin between each cell

XMARGIN = YMARGIN = 10


UP = 'up'
DOWN = 'down'
LEFT = 'left'
RIGHT = 'right'

class Node():
    def __init__(self,state,parent,action,depth,step_cost,path_cost,heuristic_cost):
        self.state = state 
        self.parent = parent # parent node
        self.action = action # move up, left, down, right      
        self.step_cost = step_cost #the tile that has been moved, the path
        self.g = depth # depth of the node in the tree
        self.f = path_cost # accumulated g(n), the cost to reach the current node
        self.h = heuristic_cost # h(n), cost to reach goal state from the current node
        # children node
        self.move_up = None 
        self.move_left = None
        self.move_down = None
        self.move_right = None
        
    def __repr__(self):
      return '\n{}, parent={}, heuristic={},f_score={}'.format(self.state,self.parent,self.h,self.f)

    def get_state(self) :
        return self.state
    
    def get_blank_position(self):
        """
        Returns the index of the blank space.
        """
        return tuple([index[0] for index in np.where(self.state == 0)])
    
    def is_valid_move(self,state, move):
        """Returns a boolean telling if the move is valid or not.
        UP, DOWN -> x dependant
        LEFT,RIGHT -> y dependant.  
        """
        blankx, blanky = self.get_blank_position()
        return (move == UP and blankx != len(state) - 1 ) or \
                 (move == DOWN and blankx != 0) or \
                 (move == LEFT and blanky != len(state) - 1) or \
                 (move == RIGHT and blanky != 0)   
     
    def is_in_grid(self,x,y):
        """Checks if the given tuple (x,y) is in the grid."""
        if x < 0 or y < 0 or x >= self.state.shape[0] or y >= self.state.shape[1]:
            return False
        return True            
    #Each action will lead to a new state
    def try_move_up(self):
      """Moving the blank tile up."""
      blankx, blanky = self.get_blank_position()

      # print('---------- #UP ⬆️  -----------')
      if not self.is_valid_move(self.state,UP):
        # print('Not a valid move')
        return False
      else:
        new_state = self.state.copy()
        lower_value = self.state[blankx + 1,blanky]    
        new_state[blankx,blanky] = lower_value
        new_state[blankx + 1,blanky] = 0
        return new_state,lower_value

    def try_move_down(self):
      """Moving the blank tile down."""
      blankx, blanky = self.get_blank_position()
      # print('---------- #DOWN ⬇️  -----------') 
      if not self.is_valid_move(self.state,DOWN):
        return False
      else:
        new_state= self.state.copy()  
        upper_value = self.state[blankx - 1,blanky]
        new_state[blankx,blanky] = upper_value
        new_state[blankx - 1,blanky] = 0  

        return new_state,upper_value

    def try_move_left(self):
      """Moving the blank tile left."""
      blankx, blanky = self.get_blank_position()

      # print('---------- #LEFT ⬅️  -----------')
      if not self.is_valid_move(self.state,LEFT):
        return False
      else:
        new_state= self.state.copy()
        right_value = self.state[blankx,blanky + 1]
        new_state[blankx,blanky] = right_value
        new_state[blankx,blanky + 1] = 0
        return new_state,right_value

    def try_move_right(self):
      blankx, blanky = self.get_blank_position()
      new_state= self.state.copy()   
      # print('---------- #RIGHT ➡️  -----------')
      if not self.is_valid_move(self.state,RIGHT):
        # print('Not a valid move')
        return False
      else:
        left_value = self.state[blankx,blanky - 1]
        new_state[blankx,blanky] = left_value
        new_state[blankx,blanky - 1] = 0
        return new_state,left_value
    
    # return user specified heuristic cost
    def get_h_cost(self,new_state,goal_state,heuristic_function):
        if heuristic_function == 'num_misplaced':
            return self.h_misplaced_cost(new_state,goal_state)
        elif heuristic_function == 'manhattan':
            return self.h_manhattan_cost(new_state,goal_state)
    
    # return heuristic cost: number of misplaced tiles
    def h_misplaced_cost(self,new_state,goal_state):
        cost = np.sum(new_state != goal_state)-1 # minus 1 to exclude the empty tile
        if cost > 0:
            return cost
        else:
            return 0 # when all tiles matches
    
    # return heuristic cost: sum of Manhattan distance to reach the goal state
    def h_manhattan_cost(self,new_state,goal_state):
        current = new_state
        # digit and coordinates they are supposed to be
        goal_position_dic = {1:(0,0),2:(0,1),3:(0,2),8:(1,0),0:(1,1),4:(1,2),7:(2,0),6:(2,1),5:(2,2)} 
        sum_manhattan = 0
        for i in range(3):
            for j in range(3):
                if current[i,j] != 0:
                    sum_manhattan += sum(abs(a-b) for a,b in zip((i,j), goal_position_dic[current[i,j]]))
        return sum_manhattan

    def get_tile_position(self,value):
        """Returns the position (x,y) for the given value of the tile."""
        return tuple([index[0] for index in np.where(self.state == value)])
      
    def get_neighbors_around_blank(self) -> list: 
        """Returns the neighbors of the blank tile."""
        bx,by = self.get_blank_position()
        around_blank_tiles = [(bx-1,by),(bx,by+1),(bx,by-1),(bx+1,by)]
        return [tile for tile in around_blank_tiles if self.is_in_grid(tile[0],tile[1])]      
    
    
    # once the goal node is found, trace back to the root node and print out the path   
    def print_path(self):
        # create FILO stacks to place the trace
        full_path = []
        state_trace = [self.state]
        action_trace = [self.action]
        depth_trace = [self.g]
        step_cost_trace = [self.step_cost]
        path_cost_trace = [self.f]
        heuristic_cost_trace = [self.h]
        
        # add node information as tracing back up the tree
        while self.parent:
            state_trace.append(self.state)
            action_trace.append(self.action)
            depth_trace.append(self.g)
            step_cost_trace.append(self.get_tile_position(self.step_cost)) #the tile that has moved
            path_cost_trace.append(self.f)
            heuristic_cost_trace.append(self.h)
            if self.action is not None:
                full_path.append((self.step_cost,self.action))
            self = self.parent
        # print out the path
        step_counter = 0
        while state_trace:
            depth = depth_trace.pop()
            h = heuristic_cost_trace.pop()
            print('step ',step_counter)
            print(state_trace.pop())
            print("""action= {}, 
                  depth= {},
                  moved tile= {},
                  h = {},
                  f_score= {}""".format(action_trace.pop(),
                                        str(depth),
                                        str(step_cost_trace.pop()),
                                        str(h),
                                        str(depth+h)
                                        )
                )
            step_counter += 1
        
        
        return full_path[::-1]
                
                        
    # search based on path cost + heuristic cost
    def a_star_search(self,goal_state,heuristic_function) -> list:
        """
        The implementation of A* Algorithm involves maintaining two lists- OPEN and CLOSED.
        OPEN contains those nodes that have been evaluated by the heuristic function but have not been expanded into successors yet. OPEN is a priority queue.

        OPEN stores nodes in the form of(priority,Node) with priority = f(n) : (f(n),n).

        CLOSED contains those nodes that have already been visited,
        to avoid repeated state, which is represented as a tuple.
        We mesure max number of nodes in the queue, measuring space performance.

        path_cost_queue represents the tiles that have been moved to reconstruct the optimal path to reach the goal state. 
        depth_queue is a queue associated to the level of the node (g(n),f(n))
        When the goal state is found, trace back to the root node and print out the path.
        """
        start = time.time()
        
        open = [(self,0)] # queue of (found but unvisited nodes, path cost+heuristic cost), ordered by the second element
        queue_num_nodes_popped = 0 # number of nodes popped off the open, measuring time performance
        queue_max_length = 1 # max number of nodes in the open, measuring space performance
        
        depth_queue = [(0,0)] # queue of node depth, (depth, path_cost+heuristic cost)
        path_cost_queue = [(0,0)] # queue for path cost, (path_cost, path_cost+heuristic cost)
        closed = set([]) # record visited states
        
        while open:
            # sort queue based on path_cost+heuristic cost, in ascending order
            open = sorted(open, key=lambda x: x[1])
            depth_queue = sorted(depth_queue, key=lambda x: x[1])
            path_cost_queue = sorted(path_cost_queue, key=lambda x: x[1])
            
            # update maximum length of the queue
            if len(open) > queue_max_length:
                queue_max_length = len(open)
                
            current_node = open.pop(0)[0] # select and remove the first node in the open
            
            queue_num_nodes_popped += 1 
            current_depth = depth_queue.pop(0)[0] # select and remove the depth for current node
            current_path_cost = path_cost_queue.pop(0)[0] # select and remove the path cost for reaching current node
            closed.add(tuple(current_node.state.reshape(1,9)[0])) # avoid repeated state, which is represented as a tuple
            
            # when the goal state is found, trace back to the root node and print out the path
            if np.array_equal(current_node.state,goal_state):
                optimal_path = current_node.print_path()
                
                print('Time performance:',str(queue_num_nodes_popped),'nodes popped off the queue.')
                print('Space performance:', str(queue_max_length),'nodes in the queue at its max.')
                print('Time spent: %0.2fs' % (time.time()-start))
                return optimal_path
            
            #We compute children 
            else:     
                # see if moving upper tile down is a valid move
                if current_node.try_move_down():
                    new_state,up_value = current_node.try_move_down()
                    # check if the resulting node is already visited
                    if tuple(new_state.reshape(1,9)[0]) not in closed:
                      
                        path_cost=current_path_cost+up_value   
                        depth = current_depth+1
                        # get heuristic cost
                        h_cost = self.get_h_cost(new_state,goal_state,heuristic_function)
                        # create a new child node
                        f_score = depth+h_cost
                        current_node.move_down = Node(state=new_state,parent=current_node,action='down',depth=depth,
                                              step_cost=up_value,path_cost=path_cost,heuristic_cost=h_cost)
                        
                        open.append((current_node.move_down, f_score))
                        depth_queue.append((depth, f_score))
                        path_cost_queue.append((path_cost, f_score))
                    
                # see if moving left tile to the right is a valid move
                if current_node.try_move_right():
                    new_state,left_value = current_node.try_move_right()
                    # check if the resulting node is already visited
                    if tuple(new_state.reshape(1,9)[0]) not in closed:
                        path_cost=current_path_cost+left_value
                        depth = current_depth+1
                        # get heuristic cost
                        h_cost = self.get_h_cost(new_state,goal_state,heuristic_function)
                        # create a new child node
                        f_score = depth+h_cost
                        current_node.move_right = Node(state=new_state,parent=current_node,action='right',depth=depth,
                                              step_cost=left_value,path_cost=path_cost,heuristic_cost=h_cost)
                        open.append((current_node.move_right, f_score))
                        depth_queue.append((depth, f_score))
                        path_cost_queue.append((path_cost, f_score))
                    
                # see if moving lower tile up is a valid move
                if current_node.try_move_up():
                    new_state,lower_value = current_node.try_move_up()
                    # check if the resulting node is already visited
                    if tuple(new_state.reshape(1,9)[0]) not in closed:
                        path_cost=current_path_cost+lower_value
                        depth = current_depth+1
                        # get heuristic cost
                        h_cost = self.get_h_cost(new_state,goal_state,heuristic_function)
                        # create a new child node
                        f_score = depth+h_cost
                        current_node.move_up = Node(state=new_state,parent=current_node,action='up',depth=depth,
                                              step_cost=lower_value,path_cost=path_cost,heuristic_cost=h_cost)
                        open.append((current_node.move_up, f_score))
                        depth_queue.append((depth, f_score))
                        path_cost_queue.append((path_cost, f_score))

                # see if moving right tile to the left is a valid move
                if current_node.try_move_left():
                    new_state,right_value = current_node.try_move_left()
                    # check if the resulting node is already visited
                    if tuple(new_state.reshape(1,9)[0]) not in closed:
                        path_cost=current_path_cost+right_value
                        depth = current_depth+1
                        # get heuristic cost
                        h_cost = self.get_h_cost(new_state,goal_state,heuristic_function)
                        # create a new child node
                        f_score = depth+h_cost
                        current_node.move_left = Node(state=new_state,parent=current_node,action='left',depth=depth,
                                              step_cost=right_value,path_cost=path_cost,heuristic_cost=h_cost)
                        open.append((current_node.move_left, f_score))
                        depth_queue.append((depth, f_score))
                        path_cost_queue.append((path_cost, f_score))

    
    
class SlidePuzzle:
    def __init__(self,initial_state:Node) -> None:
        self.grid = initial_state.state
        self.node = initial_state
        self.tiles = [(x,y)  for x in range(self.grid.shape[0]) for y in range(self.grid.shape[1])]
        self.tile_positions = {(x,y):(x*(TILESIZE+XMARGIN)+XMARGIN,y*(TILESIZE+YMARGIN)+YMARGIN) for y in range(self.grid.shape[1]) for x in range(self.grid.shape[0])}
        self.font = pygame.font.Font('freesansbold.ttf', 100)
        self.blank_position = self.node.get_blank_position()     
    
    def switch(self,tile):
        self.grid[self.blank_position],self.grid[tile] = self.grid[tile],self.grid[self.blank_position]
        self.blank_position = self.node.get_blank_position()
        
        
    def update(self,screen,dt):
        mouse = pygame.mouse.get_pressed()
        mouse_pos = pygame.mouse.get_pos()
        blankx,blanky = self.node.get_blank_position()
        slideTo = None
        if mouse[0]:
            x,y = mouse_pos[1]%(TILESIZE+XMARGIN),mouse_pos[0]%(TILESIZE+YMARGIN)
            if x > XMARGIN and y > YMARGIN:
                tile = mouse_pos[1]//TILESIZE,mouse_pos[0]//TILESIZE
                around_blank = self.node.get_neighbors_around_blank()
                
                if self.node.is_in_grid(tile[0],tile[1]) and tile in around_blank:
                    self.switch(tile)
                    if tile[0] == blankx + 1 and tile[1] == blanky:
                        slideTo = UP
                    elif tile[0] == blankx - 1 and tile[1] == blanky:
                        slideTo = DOWN
                    elif tile[0] == blankx and tile[1] == blanky + 1:
                        slideTo = LEFT
                    elif tile[0] == blankx and tile[1] == blanky - 1:
                        slideTo = RIGHT
        if slideTo:
            print(slideTo)
            
    def is_valid(self,x,y):
        """
        Checks if the move is valid and the moved tile is in the grid.
        """
        if x < 0 or y < 0 or x >= 3 or y >= 3:
            return False   
        return True
             
    def move_tile(self,direction):
        """
        Switchs the blank tile with the given direction.
        """
        blankx, blanky = self.node.get_blank_position()
        
        if direction == UP and self.is_valid(blankx+1,blanky):     
            self.switch(tile=(blankx+1,blanky))     
             
        elif direction == DOWN and blankx != 0:
            self.switch(tile=(blankx-1,blanky))
            
        elif direction == LEFT and self.is_valid(blankx,blanky+1):
            self.switch(tile=(blankx,blanky+1))   
                   
        elif direction == RIGHT and self.is_valid(blankx,blanky-1):
                self.switch(tile=(blankx,blanky-1))     
        else:
            return False
            
        return True
             
    def draw_board(self,screen, adjx=0, adjy=0):
        # draw a tile at board coordinates tilex and tiley, optionally a few
        # pixels over (determined by adjx and adjy)
        for i in range(self.grid.size):
            left, top = self.tile_positions[self.tiles[i]]
            i,j = self.tiles[i]
            if self.grid[j,i] == 0:
                color = (196, 178, 128)
                continue
            else:
                color = WHITE
            pygame.draw.rect(screen, TILE_COLOR, (left + adjx, top + adjy, TILESIZE, TILESIZE))
            pygame.draw.rect(screen, (196, 178, 128), (left + adjx, top+adjy, TILESIZE-10, TILESIZE-10))
            text = self.font.render(str(self.grid[j,i]),True,color) 
            textRect = text.get_rect()
            textRect.center = left + int(TILESIZE / 2) + adjx,top + int(TILESIZE / 2) + adjy
            screen.blit(text,textRect)

            
def main():
    global FPSCLOCK, DISPLAYSURF, BASICFONT
    pygame.init()
    
    # initial_state_tp = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    # goal_state_tp = np.array([3,4,7,5,0,8,1,2,6]).reshape(3,3)
    
    initial_state = np.array([2,8,3,1,6,4,7,0,5]).reshape(3,3)
    goal_state = np.array([1,2,3,8,0,4,7,6,5]).reshape(3,3)
    
    
    root_node = Node(state=initial_state,parent=None,action=None,depth=0,step_cost=0,path_cost=0,heuristic_cost=0)
    
    path = root_node.a_star_search(goal_state,heuristic_function = 'num_misplaced')
    
    
    pygame.display.set_caption('A* slide solver')
    screen = pygame.display.set_mode((WINDOWWIDTH,WINDOWHEIGHT))
    FPSCLOCK = pygame.time.Clock()
    
    slide_puzzle = SlidePuzzle(initial_state=root_node)
    
    
    moves = []
    for value,direction in path:
        print(str(value)+ '->' + direction)
        if direction == 'down':
            moves.append(DOWN)
        if direction == 'up':
            moves.append(UP)
            
        if direction == 'left':
            moves.append(LEFT)
            
        if direction == 'right':
            moves.append(RIGHT)
            
    print(moves)
        
        
    game_loop = True
    

    
    while game_loop:

        dt = FPSCLOCK.tick()/1000
        screen.fill(BEAUTIFUL_BLUE)
        slide_puzzle.draw_board(screen)   
        pygame.display.flip()
        
        
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()         
        

  
            
        slide_puzzle.update(screen,dt)
        if np.array_equal(slide_puzzle.node.state,goal_state):
          print('YEEY') 
          pygame.time.wait(5000)
          game_loop = False
          
          
 
        

        
        
main()

