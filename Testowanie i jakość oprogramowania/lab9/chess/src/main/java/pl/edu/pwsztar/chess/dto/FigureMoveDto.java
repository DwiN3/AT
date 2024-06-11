package pl.edu.pwsztar.chess.dto;

import pl.edu.pwsztar.chess.domain.Point;

import java.util.Map;

public record FigureMoveDto(String source, String destination, FigureType figureType) {
    private static Map<Character, Integer> loadSignFields() {
        return Map.of('a', 1, 'b', 2, 'c', 3, 'd', 4,
                'e', 5, 'f', 6, 'g', 7, 'h', 8);
    }

    private static Map<Character, Integer> loadNumberFields() {
        return Map.of('1', 1, '2', 2, '3', 3, '4', 4,
                '5', 5, '6', 6, '7', 7, '8', 8);
    }

    public Point convertToSourcePoint() {
        Character xAsChar = this.source.charAt(0);
        Character yAsChar = this.source.charAt(2);

        int x = loadSignFields().get(xAsChar);
        int y = loadNumberFields().get(yAsChar);

        return new Point(x,y);
    }

    public Point convertToDestinationPoint() {
        Character xAsChar = this.destination.charAt(0);
        Character yAsChar = this.destination.charAt(2);

        int x = loadSignFields().get(xAsChar);
        int y = loadNumberFields().get(yAsChar);

        return new Point(x,y);
    }
}
