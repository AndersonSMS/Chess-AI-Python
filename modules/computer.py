from modules.board import *
import copy

def matrix_to_tuple(array, empty_array):
    """
    Given a 2D list, converts it to 2D tuple. This is useful for using a
    matrix as a key in a dictionary
    (an empty 8x8 should be provided, just for efficiency)
    """
    for i in range(8):
        empty_array[i] = tuple(array[i])
    return tuple(empty_array)

def check_castling(board,c,side):
    '''
    Checks if castling is possible, given a board state, a color, and the side
    for the castle.
    '''
    castleLeft = False
    castleRight = False

    if c == "w":
        king = board.white_king
        leftRook = board.white_rook_left
        rightRook =  board.white_rook_right
        attacked = move_gen(board, "b", True)
        row = 7
    elif c == "b":
        king = board.black_king
        leftRook = board.black_rook_left
        rightRook =  board.black_rook_right
        attacked = move_gen(board, "w", True)
        row = 0

    squares = set()

    if king.moved == False: # cannot castle if the king has moved
        # left castle, check to see if the rook has moved
        if board.array[row][0] == leftRook and leftRook.moved == False:
            #squares between the rook and the king have to be empty and cannot be in check
            squares = {(row,1),(row,2),(row,3)}
            if not board.array[row][1] and not board.array[row][2] and not board.array[row][3]:
                if not attacked.intersection(squares):
                    castleLeft = True
        # right castle
        if board.array[row][7] == rightRook and rightRook.moved == False:
            #squares between the rook and the king have to be empty and cannot be in check
            squares = {(row,6),(row,5)}
            if not board.array[row][6] and not board.array[row][5]:
                if not attacked.intersection(squares):
                    castleRight = True

    if side == "r":
        return castleRight
    elif side == "l":
        return castleLeft

def special_move_gen(board,color,moves = None):
    '''
    From a board state and a color, returns a move dict with the possible
    special moves. Currently only returns castling moves as pawn promotion is
    implemented in a different way.

    Key in the moves dict is where the player has to 'click' to perform the move.
    Value is the special move code.
    '''
    if moves == None:
        moves = dict()
    if color == "w":
        x = 7
    elif color == "b":
        x = 0
    rightCastle = check_castling(board,color,"r")
    leftCastle = check_castling(board,color,"l")

    if rightCastle:
        moves[(x,6)] = "CR"
    if leftCastle:
        moves[(x,2)] = "CL"

    return moves


def move_gen(board, color, attc = False):
    """
    Generates the pseudo-legal moves from a board state, for a specific color.
    Does not check to see if the move puts you in check, this must be done
    outside of the function.
    Returns:
    attc = False: moves (dict) - maps coord (y,x) to a set containing the coords of
                                where it can legally move
    attc = True: moves (set) - the set of attacked squares for that color.
    """
    if attc:
        moves = set()
    else:
        moves = dict()

    # Generates all the legal moves for all the pieces, then combines them
    for j in range(8):
        for i in range(8):
            piece = board.array[i][j]
            if piece != None and piece.color == color:
                legal_moves = piece.gen_legal_moves(board)
                if legal_moves and not attc:
                    moves[(i,j)] = legal_moves
                elif legal_moves and attc:
                    moves = moves.union(legal_moves)

    return moves

# IF FUNCTION RETURNS value= -INF (or move = 0), AI IS IN CHECKMATE
# (returning +inf for value indicates player checkmate)
def minimax(board, depth, alpha, beta, maximizing, memo):
    """
    Minimax algorithm with alpha-beta pruning determines the best move for
    black from the current board state.
    Returns: bestValue - score of the board resulting from the best move
            move - tuple containing the start coord and the end coord of the best move
            ex. ((y1,x1),(y2,x2)) -> the piece at (y1,x1) should move to (y2,x2)

    Note: 0 is used as a placeholder when returning from the function, when we
    don't care about the move (eg. the algorithm is exploring options, don't
    need to return a 'move')
    """

    # convert the 2D list to a tuple, so it can be used as a key in memo
    tuple_mat = matrix_to_tuple(board.array, board.empty)

    if tuple_mat in memo and depth != 3: # set this to the depth of the initial call
        return memo[tuple_mat], 0

    if depth == 0: # end of the search is reached
        memo[tuple_mat] = board.score
        return board.score, 0

    if maximizing:
        bestValue = float("-inf")
        black_moves = move_gen(board,"b")

        # explore all the potential moves from this board state
        for start, move_set in black_moves.items():
            for end in move_set:

                # perform the move
                # preserve the start and the end pieces, in case the move
                # needs to be reversed
                piece = board.array[start[0]][start[1]]
                dest = board.array[end[0]][end[1]]

                # if a pawn promotion occurs, return the pieces involved
                pawn_promotion = board.move_piece(piece,end[0],end[1],False)

                # see if the move puts you in check
                attacked = move_gen(board,"w",True) #return spaces attacked by white
                if (board.black_king.y,board.black_king.x) in attacked:
                    # reverse the move
                    board.move_piece(piece,start[0],start[1],False, True)
                    board.array[end[0]][end[1]] = dest
                    if pawn_promotion:
                        board.score -= 9 # revert the score from the promotion
                    continue # the move is illegal, thus we don't care and move on

                #change the score if a piece was captured
                if dest != None:
                    board.score += board.pvalue_dict[type(dest)]

                # search deeper for the children, this time its the minimizing
                # player's turn
                v, __ = minimax(board, depth - 1,alpha,beta, False, memo)

                # revert the board and the score
                board.move_piece(piece,start[0],start[1],False, True)
                board.array[end[0]][end[1]] = dest
                if pawn_promotion:
                    board.score -= 9
                if dest != None:
                    board.score -= board.pvalue_dict[type(dest)]

                if v >= bestValue: # move is better than best, store it
                    move = (start, (end[0],end[1]))

                bestValue = max(bestValue, v)
                alpha = max(alpha, bestValue)

                if beta <= alpha:
                    return bestValue, move
        try:
            return bestValue, move
        except:
            return bestValue, 0 # no best move was found, indicates AI in checkmate


    else:    #(* minimizing player *)
        bestValue = float("inf")
        white_moves = move_gen(board,"w")

        # explore all the potential moves from this board state
        for start, move_set in white_moves.items():
            for end in move_set:

                # perform the move
                piece = board.array[start[0]][start[1]]
                dest = board.array[end[0]][end[1]]
                pawn_promotion = board.move_piece(piece,end[0],end[1],False)

                # see if the move puts you in check
                attacked = move_gen(board,"b",True) #return spaces attacked by white
                if (board.white_king.y,board.white_king.x) in attacked:
                    board.move_piece(piece,start[0],start[1],False,True)
                    board.array[end[0]][end[1]] = dest
                    if pawn_promotion:
                        board.score += 9
                    continue # move is illegal, don't consider it

                # update the score
                if dest != None:
                    board.score -= board.pvalue_dict[type(dest)]

                v, __ = minimax(board, depth - 1,alpha,beta, True, memo)
                
                bestValue = min(v, bestValue)
                beta = min(beta,bestValue)

                # reverse the move, revert the score
                board.move_piece(piece,start[0],start[1],False,True)
                board.array[end[0]][end[1]] = dest
                if pawn_promotion:
                    board.score += 9
                if dest != None:
                    board.score += board.pvalue_dict[type(dest)]

                if beta <= alpha:
                    return bestValue, 0

        return bestValue, 0


# IF FUNCTION RETURNS value= -INF (or move = 0), AI IS IN CHECKMATE
# (returning +inf for value indicates player checkmate)
def botIA(board, iteration, alpha, beta, White, max, memo):
    """
    Minimax algorithm with alpha-beta pruning determines the best move for
    black from the current board state.
    Returns: bestValue - score of the board resulting from the best move
            move - tuple containing the start coord and the end coord of the best move
            ex. ((y1,x1),(y2,x2)) -> the piece at (y1,x1) should move to (y2,x2)

    Note: 0 is used as a placeholder when returning from the function, when we
    don't care about the move (eg. the algorithm is exploring options, don't
    need to return a 'move')
    """

    # convert the 2D list to a tuple, so it can be used as a key in memo
    tuple_mat = matrix_to_tuple(board.array, board.empty)

    if White:
        moves = move_gen(board,"w")
    else:
        moves = move_gen(board,"b")

    ###
    bestValue = float("-inf")
    bestMove = []
    listMoves = []

    # explore all the potential moves from this board state
    for start, move_set in moves.items():
        for end in move_set:

            # perform the move
            # preserve the start and the end pieces, in case the move
            # needs to be reversed
            piece = board.array[start[0]][start[1]]
            dest = board.array[end[0]][end[1]]

            # if a pawn promotion occurs, return the pieces involved
            pawn_promotion = board.move_piece(piece,end[0],end[1],False)

            if White:
                # see if the move puts you in check
                attacked = move_gen(board,White,True) #return spaces attacked by white
                if (board.white_king.y,board.white_king.x) in attacked:
                    # reverse the move
                    board.move_piece(piece,start[0],start[1],False, True)
                    board.array[end[0]][end[1]] = dest
                    if pawn_promotion:
                        board.score -= 9 # revert the score from the promotion
                    #continue # the move is illegal, thus we don't care and move on
            else:
                # see if the move puts you in check
                attacked = move_gen(board,White,True) #return spaces attacked by white
                if (board.black_king.y,board.black_king.x) in attacked:
                    # reverse the move
                    board.move_piece(piece,start[0],start[1],False, True)
                    board.array[end[0]][end[1]] = dest
                    if pawn_promotion:
                        board.score -= 9 # revert the score from the promotion
                    #continue # the move is illegal, thus we don't care and move on

            
            #change the score if a piece was captured
            if dest != None:
                if max:
                    board.score += board.pvalue_dict[type(dest)]
                else:
                    board.score -= board.pvalue_dict[type(dest)]

            # search deeper for the children, this time its the minimizing
            # player's turn
            #v, __ = minimax(board, iteration - 1,alpha,beta, not(White), memo)

            # revert the board and the score
            board.move_piece(piece,start[0],start[1],False, True)
            board.array[end[0]][end[1]] = dest

            score = 0
            if pawn_promotion:
                score = 9
            if dest != None:
                score = board.pvalue_dict[type(dest)]
            if max:
                board.score -= score
            else:
                board.score += score

            #amazena o movimento
            thisMove = (start, (end[0],end[1]))
            thisMove = score, thisMove
            listMoves.append(thisMove)

            if score > bestValue: 
                bestMove = thisMove
                bestValue = score


            #bestValue = max(bestValue, score)
            #alpha = max(alpha, bestValue)

            #if iteration <= 1:#se chegou no final da recursão retorna o melhor movimento
                #return actualMove[0], actualMove[1]

            #if beta <= alpha:
            #    return bestValue, move
    if iteration <= 1:
        return bestValue, bestMove[1]
    if len(listMoves) > 0:
        move = 0
        bestValue = float("-inf")

        for x in range(1, 10):#os 10 melhores scores
            actualMove = listMoves[0]
            for move1 in listMoves:
                if actualMove[0] < move1[0]:
                    actualMove = move1
            listMoves.remove(actualMove)
            #copyBoard = copy.copy(board)

            #setar as coordenadas corretas das peças
            startPiece = actualMove[1][0][0], actualMove[1][0][1]
            endPiece = actualMove[1][1][0], actualMove[1][1][1]

            #aplicar as alterações no Board
            piece = board.array[startPiece[0]][startPiece[1]]#start
            dest = board.array[endPiece[0]][endPiece[1]]#end

            # if a pawn promotion occurs, return the pieces involved
            pawn_promotion = board.move_piece(piece,endPiece[0],endPiece[1],False)

            #change the score if a piece was captured
            if dest != None:
                board.score += board.pvalue_dict[type(dest)]

            valueAT, moveAT = botIA(board, iteration-1, alpha, beta, not(White), not(max), memo)
            ##################valueAT pode estar com valor errado quando uma peça e capturada
            if valueAT > bestValue:
                if dest:
                    bestValue = board.pvalue_dict[type(dest)]
                else:
                    bestValue = 0
                move = actualMove

            # revert the board and the score############################### não esta revertendo
            board.move_piece(piece,startPiece[0],startPiece[1],False, True)
            board.array[endPiece[0]][endPiece[1]] = dest
            score = 0
            if pawn_promotion:
                score = 9
            if dest != None:
                score = board.pvalue_dict[type(dest)]
            board.score -= score

        return bestValue, move
        ##retornando o move errado

    else:
        return float("-inf") , 0
    try:
        return bestValue, move
    except:
        return bestValue, 0 # no best move was found, indicates AI in checkmate


##############################################
    # convert the 2D list to a tuple, so it can be used as a key in memo
    tuple_mat = matrix_to_tuple(board.array, board.empty)

    #if tuple_mat in memo and depth != 3: # set this to the depth of the initial call
    #    return memo[tuple_mat], 0

    #if depth == 0: # end of the search is reached
    #    memo[tuple_mat] = board.score
    #    return board.score, 0

    if White:
        bestValue = float("-inf")
        black_moves = move_gen(board,"b")

        # explore all the potential moves from this board state
        for start, move_set in black_moves.items():
            for end in move_set:

                # perform the move
                # preserve the start and the end pieces, in case the move
                # needs to be reversed
                piece = board.array[start[0]][start[1]]
                dest = board.array[end[0]][end[1]]

                # if a pawn promotion occurs, return the pieces involved
                pawn_promotion = board.move_piece(piece,end[0],end[1],False)

                # see if the move puts you in check
                attacked = move_gen(board,"w",True) #return spaces attacked by white
                if (board.black_king.y,board.black_king.x) in attacked:
                    # reverse the move
                    board.move_piece(piece,start[0],start[1],False, True)
                    board.array[end[0]][end[1]] = dest
                    if pawn_promotion:
                        board.score -= 9 # revert the score from the promotion
                    continue # the move is illegal, thus we don't care and move on

                #change the score if a piece was captured
                if dest != None:
                    board.score += board.pvalue_dict[type(dest)]

                # search deeper for the children, this time its the minimizing
                # player's turn
                v, __ = minimax(board, iteration - 1,alpha,beta, False, memo)

                # revert the board and the score
                board.move_piece(piece,start[0],start[1],False, True)
                board.array[end[0]][end[1]] = dest
                if pawn_promotion:
                    board.score -= 9
                if dest != None:
                    board.score -= board.pvalue_dict[type(dest)]

                if v >= bestValue: # move is better than best, store it
                    move = (start, (end[0],end[1]))

                bestValue = max(bestValue, v)
                alpha = max(alpha, bestValue)

                if beta <= alpha:
                    return bestValue, move
        try:
            return bestValue, move
        except:
            return bestValue, 0 # no best move was found, indicates AI in checkmate


    else:    #(* minimizing player *)
        bestValue = float("inf")
        white_moves = move_gen(board,"w")

        # explore all the potential moves from this board state
        for start, move_set in white_moves.items():
            for end in move_set:

                # perform the move
                piece = board.array[start[0]][start[1]]
                dest = board.array[end[0]][end[1]]
                pawn_promotion = board.move_piece(piece,end[0],end[1],False)

                # see if the move puts you in check
                attacked = move_gen(board,"b",True) #return spaces attacked by white
                if (board.white_king.y,board.white_king.x) in attacked:
                    board.move_piece(piece,start[0],start[1],False,True)
                    board.array[end[0]][end[1]] = dest
                    if pawn_promotion:
                        board.score += 9
                    continue # move is illegal, don't consider it

                # update the score
                if dest != None:
                    board.score -= board.pvalue_dict[type(dest)]

                v, __ = minimax(board, iteration - 1,alpha,beta, True, memo)
                
                bestValue = min(v, bestValue)
                beta = min(beta,bestValue)

                # reverse the move, revert the score
                board.move_piece(piece,start[0],start[1],False,True)
                board.array[end[0]][end[1]] = dest
                if pawn_promotion:
                    board.score += 9
                if dest != None:
                    board.score += board.pvalue_dict[type(dest)]

                if beta <= alpha:
                    return bestValue, 0

        return bestValue, 0



if __name__ == "__main__":

    pygame.init()
    screen = pygame.display.set_mode((800, 60 * 8))
    b = Board()
    sprites = []

    trans_table = dict()
    value, move = minimax(b,3,float("-inf"),float("inf"), True, trans_table)
    print(len(trans_table))
    print(" ")
    b.print_to_terminal()
    print(value)
    print(move)
