public class Main {
    public static void main(String[] args) {

        int populationSize = 500;
        double probability = 0.25;
        double alpha = 0.8;
        double lb_l = -10.0, lb_r= -5.0, ub_l= 5.0, ub_r= 10.0;
        double[] lb = {lb_l, lb_r};
        double[] ub = {ub_l, ub_r};
        int maxIterations = 1000;

        int mode = 0;
        // 0 - Twoja funkcja
        // 1 - Funkcja Rosenbrocka
        // 2 - Funkcja Bootha
        // 3 - Funkcja Ackleya
        // 4 - Funkcja Rastrigina

        CuckooSearch cuckooSearch = new CuckooSearch(populationSize, probability, alpha, lb, ub, maxIterations);
        cuckooSearch.run(mode);
        System.out.println(cuckooSearch.getNameFunction());
        System.out.println(cuckooSearch.getBestSolution());
        System.out.println(cuckooSearch.getFitness());
        System.out.println(cuckooSearch.getOptimum());
    }
}