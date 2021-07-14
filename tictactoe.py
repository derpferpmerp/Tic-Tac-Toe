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


# Checks if List "a" is equal to List "b", ignoring order
def eqio(a, b):
	unmatched = list(b)
	for element in a:
		try:
			unmatched.remove(element)
		except ValueError:
			return False
	return not unmatched


def creategrid(w = 3, h = 3, coords = False, pbj = False, rjson=False, rall=False):
	top = []
	middle = []
	bottom = []
	openboardjson = {}
	for x in range(w*h):
		openboardjson[str(x)] = " "
		if pbj:
			if str(x) in [kk for kk in pbj.keys()]:
				openboardjson[str(x)] = pbj[str(x)]
	openboardjsonkeys = [k for k in openboardjson.keys()]

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
		grid = re.sub(f"\\|{x}\\|",f"|{openboardjson[x]}|",grid)
	if rjson:
		return openboardjson
	elif rall:
		return [grid,openboardjson]
	else:
		return grid

def _2D_to_1D(lst,x,y,w=width):
	B = np.array([str(x) for x in lst])
	B.shape = (w,w)
	return B[x,y]


# Prompt User to Select a Tile
def ask_for_square(inputs, w=width):
	if "-v" in sys.argv[1::]:
		print(f"Available Inputs: {inputs}\nBase Inputs: {inputsbase}")
	inputsbase = [int(str(x)) for x in range(w**2)] + [" "]
	cont=False
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
		possible = [(g + x) for g in range(0,min2win)]
		if possible[0] in starting:
			lstout.append([(g + x) for g in range(0,min2win)])
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

def all_same(items):
	return all(map(lambda x: x == items[0], items))

def all_diff_equal(lst):
	diffs = [j-i for i, j in zip(lst[:-1], lst[1:])]
	alleq = all_same(diffs)
	print(diffs, alleq)
	print(f"List: {lst} | Diffs: {diffs} | All Equal: {alleq}")
	return alleq

def combinelists(a,b):
	c=[]
	for itm in a:
		c.append(itm)
	for itm2 in b:
		c.append(itm2)
	return c

def getind(lstls,itm):
	for x in range(len(lstls)):
		if itm in lstls[x]:
			return x

def sortlist(lst):
	lst2 = lst
	lst2.sort()
	lst2 = list(dict.fromkeys(lst2))
	return lst2



#def gen_win_diagonal_3(w, h, min2win=3):
#	valid = []
#	for hiter in range(h):
#		currentwlist = []
#		for witer in range(w):
#			witer += (hiter * w)
#			if witer % 2 == 0:
#				currentwlist.append(witer)
#		valid.append(currentwlist)
#	validcombs = [list(x) for x in list(itertools.product(*valid))]
#	combsfinal_unpruned = []
#	combsout = []
#	for x in validcombs:
#		ranged = [(gg - min2win) for gg in range(min2win,len(x)+1)]
#		loopedcombs = []
#		for rangeiter in ranged:
#			loopedcombs = combinelists([list(gg) for gg in list(itertools.combinations(x, min2win + rangeiter))],loopedcombs)
#
#		for comb in loopedcombs:
#			combsorted = comb
#			combsorted.sort()
#			if combsorted == comb:
#				combsfinal_unpruned.append(comb)
#
#	for itmfinal in combsfinal_unpruned:
#		if all_diff_equal(itmfinal):
#			combsout.append(itmfinal)
#	print(f"Diagonal Combs: {combsout}")
#	return combsout

def gen_win_diagonal(w, h, min2win=3, rinps=False):
	if w == 3:
		#return gen_win_diagonal_3(3,3,min2win=min2win)
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
	for chard in [kk for kk in bjson.keys()]:
		if bjson[chard] == " ":
			continue
		if bjson[chard] not in [kk2 for kk2 in dctaggr.keys()]:
			dctaggr[bjson[chard]] = [int(chard)]
		else:
			dctaggr[bjson[chard]] += [int(chard)]
	for iterkeys in range(len(dctaggr)):
		currentvalue = [vv2 for vv2 in dctaggr.values()][iterkeys]
		currentgroup = [vv2 for vv2 in dctaggr.keys()][iterkeys]
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
