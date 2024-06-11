package pl.edu.pwsztar.chess.domain;

public interface RulesOfGame {

    /**
     * Metoda zwraca true, tylko gdy przejscie z polozenia
     * source na destination w jednym ruchu jest zgodne
     * z zasadami gry w szachy
     */
    boolean isCorrectMove(Point source, Point destination);

    class Bishop implements RulesOfGame {

        @Override
        public boolean isCorrectMove(Point source, Point destination) {
            if(source.x() == destination.x() && source.y() == destination.y()) {
                return false;
            }

            return Math.abs(destination.x() - source.x()) == Math.abs(destination.y() - source.y());
        }
    }

    class Knight implements RulesOfGame {

        @Override
        public boolean isCorrectMove(Point source, Point destination) {
            int diffX = Math.abs(destination.x() - source.x());
            int diffY = Math.abs(destination.y() - source.y());

            if((diffY == 2 || diffY ==-2) && (diffX == 1 || diffX ==-1)){
                return true;
            }
            else if((diffX == 2 || diffX ==-2) && (diffY == 1 || diffY ==-1)){
                return true;
            }

            return false;
        }
    }

    class King implements RulesOfGame {

        @Override
        public boolean isCorrectMove(Point source, Point destination) {
            int diffX = Math.abs(destination.x() - source.x());
            int diffY = Math.abs(destination.y() - source.y());

            return (diffX <= 1 && diffY <= 1);
        }
    }

    class Pawn implements RulesOfGame {

        @Override
        public boolean isCorrectMove(Point source, Point destination) {
            int diffX = Math.abs(destination.x() - source.x());
            int diffY = destination.y() - source.y();

            if (diffX == 0 && diffY > 0 && diffY <= 2) {
                return  (diffY == 1 || (diffY == 2 && source.y() == 1));
            }
            return false;
        }
    }

    class Rook implements RulesOfGame {

        @Override
        public boolean isCorrectMove(Point source, Point destination) {
            int diffX = Math.abs(destination.x() - source.x());
            int diffY = destination.y() - source.y();

            if((diffX != 0 && diffY == 0) || (diffX == 0 && diffY != 0)){
                return true;
            }

            return false;
        }
    }

    class Queen implements RulesOfGame {

        @Override
        public boolean isCorrectMove(Point source, Point destination) {
            int diffX = Math.abs(destination.x() - source.x());
            int diffY = destination.y() - source.y();

            if((diffX != 0 && diffY == 0) || (diffX == 0 && diffY != 0)){
                return true;
            }
            else if(source.x() != destination.x() || source.y() != destination.y()) {
                return Math.abs(destination.x() - source.x()) == Math.abs(destination.y() - source.y());
            }
            return false;
        }
    }
}
