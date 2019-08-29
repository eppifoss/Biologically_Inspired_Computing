import csv
import sys
import time
import itertools
import random

def tour_length(tour):
    """
    Calculates the length of the tour for the given cities(tour).

    Parameters
    ----------
    tour : list
         The list of the cities for the tour.

    Returns
    -------
    length : float
           The length of the tour.
    """
    length = 0.0
    cities = len(tour)
    for x in range (cities):
        y = (x+1)% cities
        length += distance_table[tour[x]][tour[y]]

    return length

    
def create_tour(size):
    """
    Create all possible permutation of the tour.
    Used for exhaustive search.

    Parameters
    ----------
    size : int
       The size of no. of cities to create the tour for.

    Returns
    -------
    All permutations of the tour, that is iterable.
    """
    return itertools.permutations(size)

def exhaustiveSearch(size):
    """
    Exhaustive search for the best tour by checking all
    possible permutations.

    Parameters
    ----------
    size : int
      The size of no. of cities to create the tour for.

    Returns
    -------
    best_tour : tour list
          The best tour.
    """
    #Create tour for the first 'size' cities.
    tours = create_tour(xrange(size))

    best_tour_length = sys.float_info.max
    best_tour = ()                          

    for x in tours:
        dist= tour_length(x)
        if( dist < best_tour_length):
            best_tour = x
            best_tour_length = dist

    return best_tour,best_tour_length


    
# Read data file
distance_table = []

with open('european_cities.csv') as file:
    reader = csv.reader(file,delimiter=';')
    # First row is the city names
    city_names = reader.next()
    # The rest of the rows are the distance table
    for row in reader:
        distance_table.append([float(cell) for cell in row])

"""    
# List the city names
print city_names
# Print out the size of the table, should be square
print len(distance_table), len(distance_table[0])
# Look up a distance and print it
city_A = 3 # Simply refer to cities by
city_B = 5 # their index in the table
print 'The distance from',city_names[city_A],'to',city_names[city_B],'is',distance_table[city_A][city_B],'km'
"""

###############################################################
# The hill climbing algorithm and it's helping methods start here:

def create_random_tour(size):
    """
    Creates random tour of the given size. Used by hill climbing.
    Parameters
    ----------
    size : int
       The size of the tour
    """
    tour = random.sample(xrange(0,size),size)
    return tour

def all_pairs(size):
    """
    Creates all possible pairs of the given list size
    Parameters
    ----------
    size : int
        The size of the list.
    """
    r1=range(size)
    for i in r1:
        for j in r1:
            yield (i,j)
            
def swap_place(tours):
    """
    Swap the places of the elements of tours.
    Parameters
    ----------
    tours : list
         The tours/routes
    """
    for i,j in all_pairs(len(tours)):
        if i<j:
            copy = tours[:]
            copy[i], copy[j] = copy[j],copy[i]
            yield copy
            
def create_Children(tours):
    """
    Create all children of the given tour.
    The closet variations of the tours.
    Parameters
    ----------
    tours : list
       The tour/ route
    """
    children= []
    new_child = swap_place(tours)    #Create variations 
    for x in new_child:
        children.append(x)         #Record it

    return children


def hill_Climbing(size):
    """
    Hill Cllimbing search algorithm for the travelling salesman
    problem.

    Parameters
    ----------
    size : int
        The no. of cities. The size of the route.

    """
    tour = create_random_tour(size)      #Random init
    best_length = tour_length(tour)      #Calculate length
    x = 1
    max_evals = 2000 * size             
    while x < max_evals:   

        found_new= False                    
        for child in create_Children(tour):  #Check for each child
            if x >= max_evals :   
                break

            child_length = tour_length(child)    
            x += 1
            if child_length < best_length :    #if better solution
                best_length = child_length
                tour = child
                found_new = True
                break

        if (not found_new):      #If stuck at local optimum
            break
            
    return tour, best_length


######################################################
#The genetic algorithm and it's helping methods start here:

population_size = 3          #The population for a generation
max_generations = 1000      # Max no. of generations

def init_random_population(size,population_size):
    """
    Initialize random population with 'population_size' for
    'size' no. of cities.
    Parameters
    ----------
    size : int
        The no. of cities. The size of the routes.

    population_size : int
        The no. routes in a population
    """
    population = []
    for x in range(population_size):
        population.append(create_random_tour(size))
        
    return population

def fitness(population):
    """
    The fitness of a population. Calculated as 1/(tour length).
    Checks the fitness of a single route.
    Parameters
    ----------
    population : int
       A route to check the tour_length of.
    """
    return (1/tour_length(population))

def evaluate_fitness(population):
    """
    Evaluates the fitness of the entire population.
    Parameters
    ----------
    population : list
           The entire population of routes.
    """
    fitness_list = dict()

    #calculate fitness of all population 
    for x in population:    
        val = fitness(x)
        #add to dict() with fitness as the key and
        # route as the value
        fitness_list[val]= x      

    return (fitness_list)

def select_parents(fitness_list):
    """
    Select parents of the current generation based on the
    fitness of the population.
    Parameters
    ----------
    fitness_list : dict
         A dict() with fitness and population.
    """
    p1 = None
    p2 = None
    #Get the two max values and assign as parents p1 and p2
    for keys in sorted(fitness_list,reverse=True):
        if(p1 == None):
            p1 = fitness_list[keys]
        elif(p2 == None):
            p2 = fitness_list[keys]
            parents = [p1,p2]
            return parents

def recombine(parents):
    """
    Recombine the parents. Crossover between parents to create
    a child.
    Parameters
    ----------
    parents : list
        The parents; 2 max routes of a generation.
    """
    p1 = parents[0]
    p2 = parents[1]
    c = random.randrange(len(p1))  #Create random crossover point
    child = p1[:c] + p2[c:]        #Cross over
    check = []
    for x in child:               #Check of repeat no.
        if x in check:            #Don't add repeat no.
            pass
        else:
            check.append(x)

    no =  range(len(child))
    for x in no:               #Check for remaingin no.s not added
        if x in check:     
            pass
        else:
            check.append(x)     # Add the numbers left

    child = check
    return child

def mutate(child):
    """
    Mutate the child. Swaps random positions.
    Parameters
    ----------
    child : list
        The route. 
    """
    #for random no. of times 
    for x in range(random.randrange(len(child))):
        posx = random.randrange(len(child))       #random x
        posy = random.randrange(len(child))       # random y
        #Swap elements with position x and y in child.
        child[posx] , child[posy] = child[posy] ,child[posx]

    return child

def add_to_population(children,population):
    """
    Add the child(route) to the population(all the routes.)
    Parameters
    ----------
    children : list
           A route.
    population : list
          All the routes.
    """
    population.append(children)
    return population

def select_survivors(fitness_list):
    """
    Select the survivors of the population. The routes with
    the higher fitness are chosen and lower fitness are
    discarded.
    Parameters
    ----------
    fitness_list : dict
          The dict of fitness (key) and population (value).
    """
    population = []

    #Sometimes two tours have the same fitness.
    #Here I have used dict with key as a fitness.
    #This can cause error when I remove a key and its
    # corresponding value, as it removes the population
    #can become 1(causing problem for recombination)
    #So a route is removed only if the population is greater than
    # the 'population_size'
    if ( len(fitness_list) > population_size):
        remove = min(fitness_list)
        fitness_list.pop(remove,None)

    for x in fitness_list.values():
        population.append(x)

    return population

def get_best(fitness_list):
    """
    Get the best fitness value from the fitness_list.
    Parameters
    ----------
    fitness_list : dict
          The dict() of fitness (key) and population (value).
    """
    key = max(fitness_list)
    best = fitness_list[key]
    return best,key


def genetic_algorithm(size):
    """
    A genetic algorithm for the travelling salesman problem.

    Parameters
    ----------
    size : int
        The no. of cities. The size of the routes.
    """
    
    avg_fitness = []
    #Initialize random population
    population = init_random_population(size,population_size)
    fitness_list = evaluate_fitness(population)  #calc fitness

    generation = 0
    while generation < max_generations :   #For max generations
        parents = select_parents(fitness_list)   #select parents
        children = recombine(parents)       #crossover of parents
        children = mutate(children)         #mutate child
        population = add_to_population(children, population)
        fitness_list = evaluate_fitness(population)
        population = select_survivors(fitness_list)
        best_fit = get_best(fitness_list) #best of current gen.
        avg_fitness.append(best_fit[1])   #add fitness of best
        generation +=1

    avg = 0
    #Calculate avg. fitness of all generations    
    for x in avg_fitness:       
        avg += x

    avg = avg/generation
    print avg
    return best_fit[0] , tour_length(best_fit[0])


######## Other supporting functions : ############


size = 5

start = time.time()
print exhaustiveSearch(size)
end=time.time()-start
print end

start = time.time()
print hill_Climbing(size)
end=time.time()-start
print end

start = time.time()
print genetic_algorithm(size)
end=time.time()-start
print end






