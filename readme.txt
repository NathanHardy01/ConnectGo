Nathan Hardy

Bot Name: ConnectGO

Calculating the potential points of the current state was my first thought. I used a rudimentary system to calculate
that for every row, column and diagonal, one that I believe falls short on being accurate when calculating a row's potential
points. My heuristic is pretty simple but more effective than the basic utility function when added to a weighted function.

The test board I created was a specific arrangment where I was testing my algorithms' decision making given an obvious move.
It's cool because minimax picked the move that had more long term benefit than my lookahead picked.

I think all should be working.