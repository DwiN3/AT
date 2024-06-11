package pl.edu.pwsztar.chess.service.impl;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.stereotype.Service;
import pl.edu.pwsztar.chess.domain.RulesOfGame;
import pl.edu.pwsztar.chess.dto.FigureMoveDto;
import pl.edu.pwsztar.chess.service.ChessService;

@Service
public class ChessServiceImpl implements ChessService {
    private static final Logger LOGGER = LoggerFactory.getLogger(ChessServiceImpl.class);
    private final RulesOfGame bishop;
    private final RulesOfGame knight;
    private final RulesOfGame king;
    private final RulesOfGame pawn;
    private final RulesOfGame rook;
    private final RulesOfGame queen;

    public ChessServiceImpl() {
        bishop = new RulesOfGame.Bishop();
        knight = new RulesOfGame.Knight();
        king = new RulesOfGame.King();
        pawn = new RulesOfGame.Pawn();
        rook = new RulesOfGame.Rook();
        queen = new RulesOfGame.Queen();
    }

    public boolean isCorrectMove(FigureMoveDto figureMoveDto) {

        final var source = figureMoveDto.convertToSourcePoint();
        final var destination = figureMoveDto.convertToDestinationPoint();

        LOGGER.info("*** move after convert, source      : {}", source);
        LOGGER.info("*** move after convert, destination : {}", destination);

        // refaktoryzacja? switch to nie jest dobre rozwiazanie
        return switch (figureMoveDto.figureType()) {
            case BISHOP -> bishop.isCorrectMove(source, destination);
            case KNIGHT -> knight.isCorrectMove(source, destination);
            case KING -> king.isCorrectMove(source, destination);
            case PAWN -> pawn.isCorrectMove(source, destination);
            case ROOK-> rook.isCorrectMove(source, destination);
            case QUEEN-> queen.isCorrectMove(source, destination);
            default -> false;
        };
    }
}
