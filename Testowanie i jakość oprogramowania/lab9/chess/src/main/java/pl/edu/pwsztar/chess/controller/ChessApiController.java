package pl.edu.pwsztar.chess.controller;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import pl.edu.pwsztar.chess.dto.FigureMoveDto;
import pl.edu.pwsztar.chess.service.ChessService;

@Controller
@RequestMapping(value="/api")
public class ChessApiController {

    private static final Logger LOGGER = LoggerFactory.getLogger(ChessApiController.class);

    private final ChessService chessService;

    public ChessApiController(ChessService chessService) {
        this.chessService = chessService;
    }

    @PostMapping(value = "/chess/is-correct-move")
    ResponseEntity<Boolean> isCorrectMove(@RequestBody FigureMoveDto figureMoveDto) {
        LOGGER.info("*** move details : {}", figureMoveDto);

        // TODO: true = ruch dozwolony (figura moze przemiescic sie z punktu source do punktu destination)
        // TODO: false = ruch zabroniony (figura nie moze przemiescic sie z punktu source do punktu destination)
        return ResponseEntity.ok(chessService.isCorrectMove(figureMoveDto));
    }
}
