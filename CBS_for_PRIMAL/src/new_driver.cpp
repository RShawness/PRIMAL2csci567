#include <boost/program_options.hpp>
#include <boost/tokenizer.hpp>
#include "CBS.h"


list<list<pair<int, int>>> findPath_new(list<bool> map, list<int> startL, list<int> startD, list<int> goalL, int cols, int rows, int agents, int time_limit) {
    cout << "finding path in cbs" << endl;
	// (list<bool> map, list<int> startL, list<int> startD, list<int> goalL, int cols, int rows, int agents);
	Instance instance(map, startL, startD, goalL, cols, rows, agents);

    int runs = 1;
	// cout << "printing instance" << endl;

	// cout << "number of columns: " << instance.num_of_cols << endl;
	// cout << "number of rows: " << instance.num_of_rows << endl;
	// cout << "map size: " << instance.map_size << endl;
	// instance.printMap();

    // initialie the solver
	CBS cbs(instance, false, 1);
	cbs.setPrioritizeConflicts(false);
	cbs.setDisjointSplitting(false);
	cbs.setBypass(true);
	cbs.setRectangleReasoning(rectangle_strategy::GR);
	cbs.setCorridorReasoning(corridor_strategy::C);
	cbs.setHeuristicType(heuristics_type::WDG);
    cbs.setClusterHeuristicType(cluster_heuristics_type::CHBP);
	cbs.setTargetReasoning(true);
	cbs.setMutexReasoning(false);
	cbs.setSavingStats(false);
	cbs.setNodeLimit(MAX_NODES);    

	list<list<pair<int, int>>> empty_list;
    // run
    double runtime = 0;
	int min_f_val = 0;
	for (int i = 0; i < runs; i++)
	{
		cbs.clear();
		bool solution_found = cbs.solve(time_limit, min_f_val);
		if (!solution_found) {
			return empty_list;
		}
		runtime += cbs.runtime;
		if (cbs.solution_found)
			break;
		min_f_val = (int) cbs.min_f_val;
		cbs.randomRoot = true;
	}
	cbs.runtime = runtime;

    // write results to file
    return cbs.pathMatrix();

}

