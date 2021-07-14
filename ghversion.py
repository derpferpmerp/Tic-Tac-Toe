import itertools
import re
import inquirer
import numpy as np
import sys


maxlength = 12

width = int(inquirer.prompt([inquirer.List("width", message = "Length of Board?", choices = [str(rr) for rr in range(3,maxlength + 1)])])["width"])
height= width


def square(txt, t=False, m=False, b=False):
	if t:
		return "|\u203E\u203E\u203E\u203E\u203E|"
	elif m:
		return f"| |{txt}| |"
	elif b:
		return "|_____|"

def chunks(lst, n):
	for i in range(0, len(lst), n):
		yield lst[i : i + n]


def creategrid(w = 3, h = 3, coords = False, pbj = False, rjson=False, rall=False):
	top = []
	middle = []
	bottom = []
	openboardjson = {}
	for x in range(w*h):
		openboardjson[str(x)] = " "
		if pbj:
			if str(x) in list(pbj.keys()):
				openboardjson[str(x)] = pbj[str(x)]
	openboardjsonkeys = list(openboardjson.keys())

	for witer_t in range(w):
		top.append(square(" ", t=True))
		middle.append(square("{}", m=True))
		bottom.append(square(" ", b=True))
	top, middle, bottom = ["".join(top), "".join(middle), "".join(bottom)]
	grid = ""

	if not coords:
		lstmidletters = list(chunks([str(x) for x in range(w * h)], w))
	else:
		lstmidletters = list(chunks([str(x) for x in coords], w))

	for hiter in range(h):
		hitermid = lstmidletters[hiter]
		grid += "\n".join([top, middle.format(*hitermid), bottom]) + (
			["\n" if hiter != range(h)[-1] else ""][0]
		)
	for x in openboardjsonkeys:
		grid = re.sub(f"\\|{x}\\|", f"|{openboardjson[x]}|", grid)
	if rjson:
		return openboardjson
	elif rall:
		return [grid,openboardjson]
	else:
		return grid

def _2D_to_1D(lst,x,y,w=width):
	B = np.array([str(x) for x in lst])
	B.shape = (w, w)
	return B[x, y]


# Prompt User to Select a Tile
def ask_for_square(inputs, w=False):
	if "-v" in sys.argv[1::]:print(f"Available Inputs: {inputs}\nBase Inputs: {inputsbase}")
	inputsbase = [int(str(x)) for x in range(w**2)] + [" "]
	for x in inputsbase:
		if x not in list([int(g) for g in inputs]):
			inputsbase[inputsbase.index(x)] = None
	inputs = inputsbase[0:-1]
	B = np.array(inputs)
	B.shape = (w, w)
	expanded = [[B[g,x] for x in range(w)] for g in range(w)]
	lstcoords = []
	for gg in inputs:
		for itm2 in range(len(expanded)):
			if gg in expanded[itm2]:
				if int(_2D_to_1D([str(x) for x in range(w**2)],int(itm2),int(expanded[itm2].index(gg)))) in inputs:
					lstcoords.append(f"R: {expanded[itm2].index(gg)}, C: {itm2}")
				else: continue
	coordinate = [inquirer.List("Set Square", message = "Which Coordinate?", choices = lstcoords)]
	coordinatechosen = inquirer.prompt(coordinate)["Set Square"].split(",")
	for unfmt in range(len(coordinatechosen)):
		coordinatechosen[unfmt] = re.sub("[^\\d]+","",coordinatechosen[unfmt])
	coordinatechosen = str(_2D_to_1D([str(x) for x in range(w**2)],int(coordinatechosen[1]),int(coordinatechosen[0])))
	choicesforcoordinate = [inquirer.List("Set Square Value", message = "Cross or Circle?", choices = ["X","O"])]
	value2set = inquirer.prompt(choicesforcoordinate)
	return {coordinatechosen:value2set["Set Square Value"]}


# Generates all of the win combinations, in a list of lists, for horizontal combinations of length "min2win"
def gen_win_horizontal(w, h, min2win=3):
	rangeminmax = (w * h) - min2win
	lstout = []
	starting = [0]
	for pp in range(h - 1):
		starting.append(starting[-1] + w)
	for pp2 in range(1, w - min2win + 1):
		toappend = [gg + pp2 for gg in starting[0:h]]
		for itm in toappend:
			starting.append(itm)
	for x in range(rangeminmax):
		possible = [(g + x) for g in range(min2win)]
		if possible[0] in starting:
			lstout.append([(g + x) for g in range(min2win)])
	return lstout

# Generates all of the win combinations, in a list of lists, for vertical combinations of length "min2win"
def gen_win_vertical(w, h, min2win=3):
	lstinit = [0]
	lstout = []
	for initialiter in range(1, h):
		lstinit.append(initialiter)
	for seconditer in range(1, w - min2win + 1):
		toappend = [gg + (w*seconditer) for gg in lstinit[0:h]]
		for itm in toappend:
			lstinit.append(itm)
	for itmpossible in lstinit:
		lstout2out = []
		for iter2 in range(min2win):
			lstout2out.append(itmpossible + (iter2 * w))
		lstout.append(lstout2out)
	return lstout

# Concats Two Lists (Before I realized that combinelists(a,b) is literally equal to a + b)
def combinelists(a,b):
	c=[]
	for itm in a:
		c.append(itm)
	for itm2 in b:
		c.append(itm2)
	return c

# Get Index of Item in List (Yes, I could use .index, but it was 3AM on a Tuesday)
def getind(lstls,itm):
	for x in range(len(lstls)):
		if itm in lstls[x]:
			return x

# Sorts List and Returns it, without duplicates, sortlist([20,3,1,3,4,5]) = [1, 3, 4, 5, 20]
def sortlist(lst):
	lst2 = lst
	lst2.sort()
	lst2 = list(dict.fromkeys(lst2))
	return lst2



# 	Generates all of the win combinations in the diagonal direction.
# Basically Creates A Checkerboard pattern, (Squares [1,3,5,7], then
# Squares [2,4,6,8] for a 3x3 grid.) takes that and converts it to coordinates,
# Literally as simple as turning [0,0] into 0, and [2,2] (for a 3x3 grid)
# into 8 (0,1,2) (Group 1), (3,4,5) (Group 2), (XAxis + 2) (3,4,5) -> (6,7,8)
# All it entails is using numpy to create a 2D Array, display it to the user,
# Have them return a guess, use regex to convert that back into a string,
# Then convert the string back into the coordinate system, then convert the 2D
# Data into 1D, [0,0] -> Slot 0:
# 						-----
# 						0 1 2
# 						3 4 5
#						6 7 8
# 						-----
# 	For a 3x3 Grid, where the numbers represent the ID. Well anyways, it then
# returns a group of 3, where the group can be anything, as long as it is paired
# With another coordinate that is "min2win" distance away from it, based on
# The Cartesian System (meaning delta([1,1]) - delta([0,0]) == 1), and if the 
# amount of combinations is greater than 3, it will measure the delta from the
# Width and min2win (if w = 5, delta=2 from defaults), meaning that it will
# look for combinations with a distance of one, then a distance of two, and use
# the sortlist() function to sort the lists by ascending order (trick used so
# that I know it won't loop back, (for example, referencing the image above,
# It would consider a move like [0,4,6] to be valid, if I didn't give the
# Ascending order rule)). So now that all of that is out of the way, to sum
# it up:
# 
# ------ gen_win_diagonal(w (width), h (height), min2win (minimum length combo), rinps (dev feature)) ------
# - Converts range data into Numpy Array
# - Sorts it into Evens and Odds (Creating Checkerboard Pattern)
# - Puts the numpy array into "list of lists" format
# - Parses the list and converts it into a list of lists: [character, [row,col]]
# - Converts that List into a JSON Dictionary
# - Opens the list and looks for another term that is distance (min2win, if == 3, else iterable that is range(min2win - 3))
#   between the row value, and appends that to a list "groups"
# - The list "groups" gets combined using the sortlist() function, allowing the list to be sorted in
#   ascending order, along with removing duplicates in the list. It then breaks up the list into chunks based on the
#   min2win - 1 (Derived from human observation), and then outputs verbose dev output (toggle with python3 tictactoe.py [-p/None])
#   and finally returns the chunked output to the User.


def gen_win_diagonal(w, h, min2win=3, rinps=False):
	if w == 3:
		return [[0, 4, 8], [2, 4, 6]]
	ml = min2win
	s = w
	if ml > s: return []
	B = np.array([[i] for i in [int(str(x)) for x in range(s**2)]])
	B.shape = (s, s)
	odds, evens = [[],[]]
	for num in range(s):
		if num % 2 == 0:
			evens.append(num)
		else:
			odds.append(num)
	lstarr = []
	for itr in range(s):
		lstarr.append(evens)
		lstarr.append(odds)
	lstarr = lstarr[0:s]
	rows = [[B[g,x] for x in range(s)] for g in range(s)]
	for itrind in range(s):
		lstarr[itrind] = [{x + (s * itrind):[getind(rows,(x + (s * itrind))),rows[getind(rows,(x + (s * itrind)))].index(x + (s * itrind))]} for x in lstarr[itrind]]   
	lstarr = {k:v for x in sum(lstarr,[]) for k,v in x.items()}
	if rinps:
		return [f"{list(x[::-1])[0]}{list(x[::-1])[1]}" for x in list(lstarr.values())]
	groups = []
	for itm in range(len(lstarr)):
		if itm != len(lstarr) - 1:
			for itr3 in range(itm,len(lstarr)-1):
				try:
					for mliter in range(2,ml+1):
						if list(lstarr.values())[itr3][0] == list(lstarr.values())[itm][0] + (mliter-1):
							if abs(list(lstarr.values())[itr3][1]-list(lstarr.values())[itm][1]) == (mliter-1):
								groups.append([list(lstarr.keys())[itm],list(lstarr.keys())[itm + itr3]])
				except IndexError:
					continue
	amt2merge = abs(ml - 1)
	groups = [sortlist(g) for g in [sum(x,[]) for x in list(chunks(groups, amt2merge))]]
	if "-v" in sys.argv[1::]:
		print(f"Diagonal Combs: {groups}")
	return groups



def gen_win_conditions(wnum, hnum, m2w=3):
	combsout = []
	horizontal = gen_win_horizontal(wnum, hnum, min2win=m2w)
	vertical = gen_win_vertical(wnum, hnum, min2win=m2w)
	diagonal = gen_win_diagonal(wnum, hnum, min2win=m2w)
	for combiterh in horizontal:
		combsout.append(combiterh)
	for combiterv in vertical:
		combsout.append(combiterv)
	for combiterd in diagonal:
		combsout.append(combiterd)
	return combsout


def all_in(lstin, lstcheck):
	for itmcheck in lstin:
		if itmcheck not in lstcheck:
			return False
	return True


def detect_win(wconds,bjson):
	dctaggr = {}
	for chard in list(bjson.keys()):
		if bjson[chard] == " ":
			continue
		if bjson[chard] not in list(dctaggr.keys()):
			dctaggr[bjson[chard]] = [int(chard)]
		else:
			dctaggr[bjson[chard]] += [int(chard)]
	for iterkeys in range(len(dctaggr)):
		currentvalue = list(dctaggr.values())[iterkeys]
		currentgroup = list(dctaggr.keys())[iterkeys]
		for wcond in wconds:
			if all_in(wcond, currentvalue):
				return currentgroup
	return False



# ------------------------------------------------------------------
# ------------------------------------------------------------------
# ------------------------------------------------------------------

def rungame(w_ingame,h_ingame):
	minimum2win = int(inquirer.prompt([inquirer.List("min2winjson", message = "Minimum Combination Length?", choices = [str(rr) for rr in range(3,w_ingame + 1)])])["min2winjson"])
	winconditions = gen_win_conditions(w_ingame,h_ingame,m2w=minimum2win)
	print(
		creategrid(
			w = w_ingame,
			h = h_ingame,
			coords = [str(x) for x in range(w_ingame * h_ingame)],
		),
	)

	newselected = ask_for_square([str(x) for x in range(w_ingame * h_ingame)])
	grout,obj = creategrid(w = w_ingame, h = h_ingame, pbj = newselected, rall=True)
	print(grout)
	newselected2 = ask_for_square([str(x) for x in range(w_ingame * h_ingame) if x not in [int(k) for k in newselected.keys()]])
	newselected[[kk for kk in newselected2.keys()][0]] = [vv for vv in newselected2.values()][0]
	grout,obj = creategrid(w = w_ingame, h = h_ingame, pbj = newselected, rall=True)
	print(grout)
	gameend = False
	while gameend == False:
		newselected2 = ask_for_square([str(x) for x in range(w_ingame * h_ingame) if x not in [int(k) for k in newselected.keys()]])
		newselected[[kk for kk in newselected2.keys()][0]] = [vv for vv in newselected2.values()][0]

		grout,obj = creategrid(w = w_ingame, h = h_ingame, pbj = newselected, rall=True)
		print(grout)
		if detect_win(winconditions,obj) != False:
			print(f"Group {detect_win(winconditions,obj)} has won!")
			gameend = True

# ------------------------
rungame(width,height)
