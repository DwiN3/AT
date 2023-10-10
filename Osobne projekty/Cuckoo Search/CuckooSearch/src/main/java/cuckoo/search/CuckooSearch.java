package cuckoo.search;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public class CuckooSearch {

    private int populationSize; // Liczba potencjalnych rozwiązań problemu
    private double probability; // Prawdopodobieństwo z jakim kukułki będą próbowały zastąpić gniazda innych kukułek
    private double alpha; // Skala kroku, którym poruszają się kukułki
    private double[] lowerBorder; // Dolna granica przedziału poszukiwań optymalnego rozwiązania
    private double[] upperBorder; // Górna granica przedziału poszukiwań optymalnego rozwiązania
    private int maxIterations; // Maksymalna liczba iteracji algorytmu
    private Random random; // Obiekt generatora liczb losowych

    private String nameFunction="Twoja Funkcja"; // Przechowywanie nazwy funkcji
    private String bestSolution="", optimum=""; // Przechowywanie najlepszego rozwiązania i optimum
    private double fitness=0; // Przechowywanie najleps
    private int mode=0; // Przechowywanie trybu
    private int count=10; // Licznik wyświetlający najlepsze rozwiązania

    public String getNameFunction() {
        return nameFunction;
    }

    public String getBestSolution() {
        return bestSolution;
    }

    public double getFitness() {
        return fitness;
    }

    public String getOptimum() {
        return optimum;
    }

    public CuckooSearch(int populationSize, double probability, double alpha, double[] lowerBorder, double[] upperBorder, int maxIterations) {
        this.populationSize = populationSize;
        this.probability = probability;
        this.alpha = alpha;
        this.lowerBorder = lowerBorder;
        this.upperBorder = upperBorder;
        this.maxIterations = maxIterations;
        this.random = new Random(); // Inicjalizacja generatora liczb losowych
    }

    public double[] run() {
        // Inicjalizacja populacji
        double[][] population = new double[populationSize][lowerBorder.length];
        for (int i = 0; i < populationSize; i++) {
            for (int j = 0; j < lowerBorder.length; j++) {
                population[i][j] = lowerBorder[j] + random.nextDouble() * (upperBorder[j] - lowerBorder[j]);
            }
        }

        // Sortowanie populacji według wartości funkcji celu
        sortPopulation(population);

        // Inicjalizacja najlepszego rozwiązania
        double[] bestSolution = population[0];

        // Wykonywanie iteracji
        for (int iteration = 1; iteration <= maxIterations; iteration++) {
            // Generowanie nowej populacji
            double[][] newPopulation = new double[populationSize][lowerBorder.length];
            for (int i = 0; i < populationSize; i++) {
                // Generowanie nowego rozwiązania
                double[] newSolution = new double[lowerBorder.length];
                for (int j = 0; j < lowerBorder.length; j++) {
                    double r = random.nextDouble();
                    if (r < probability) {
                        // Generowanie nowego rozwiązania za pomocą lotów Levy'ego
                        double sigma = Math.pow((gamma(1.0 + alpha) * Math.sin(Math.PI * alpha / 2.0) / (gamma((1.0 + alpha) / 2.0) * alpha * Math.pow(2.0, (alpha - 1.0) / 2.0))), 1.0 / alpha);
                        double u = random.nextGaussian() * sigma;
                        double v = random.nextGaussian() * sigma;
                        double step = u / Math.pow(Math.abs(v), 1.0 / alpha);
                        double newValue = population[i][j] + step;
                        newSolution[j] = clamp(newValue, lowerBorder[j], upperBorder[j]);
                    } else {
                        // Generowanie nowego rozwiązania za pomocą losowego przesunięcia
                        int cuckooIndex = random.nextInt(populationSize);
                        double[] cuckoo = population[cuckooIndex];
                        double stepSize = random.nextDouble() * alpha;
                        double[] step = new double[lowerBorder.length];
                        for (int k = 0; k < lowerBorder.length; k++) {
                            step[k] = stepSize * (cuckoo[k] - population[i][k]);
                        }
                        double[] newValue = new double[lowerBorder.length];  // obliczanie wartości funkcji celu dla nowego rozwiązania
                        for (int k = 0; k < lowerBorder.length; k++) {
                            newValue[k] = population[i][k] + step[k];
                        }
                        newSolution[j] = clamp(newValue[j], lowerBorder[j], upperBorder[j]);
                    }
                }
                newPopulation[i] = newSolution;
            }

            //
            sortPopulation(newPopulation);


            for (int i = populationSize - 1; i >= populationSize / 2; i--) {
                population[i] = newPopulation[populationSize - 1 - i];
            }

            // Sortuj populację według sprawności
            sortPopulation(population);

            // Zaktualizuj najlepsze rozwiązanie
            if (fitness(population[0]) < fitness(bestSolution)) {
                bestSolution = population[0];
            }

            // Wyświetlanie 10 przykładowych iteracji najlepszego rozwiązania (TESTY)
            count-=1;
            //if(count > 0) System.out.println("Iteration " + iteration + ": " + fitness(bestSolution));

            // Wyświetlanie iteracji najlepszego rozwiązania
            double[] currentBest = newPopulation[0];
            for (int i = 1; i < populationSize; i++) {
                if (fitness(newPopulation[i]) > fitness(currentBest)) {
                    currentBest = newPopulation[i];
                }
            }
            // Wyświetlanie przykładowych iteracji najlepszego rozwiązania (TESTY)
            //System.out.println("Best solution in population " + iteration + ": " + Arrays.toString(currentBest));
        }

        return bestSolution;
    }

    // Sortuje populację według wartości funkcji dopasowania w porządku rosnącym
    private void sortPopulation(double[][] population) {
        Arrays.sort(population, Comparator.comparingDouble(this::fitness));
    }

    // Zwraca wartość funkcji dopasowania rozwiązania
    private double fitness(double[] solution) {
        double x = solution[0];
        double y = solution[1];
        return Math.pow(x, 2) + Math.pow(y, 2);
    }

    // Zwraca losową wartość z rozkładu normalnego
    private double randn() {
        return random.nextGaussian();
    }

    // Zwraca wartość funkcji gamma dla x
    private double gamma(double x) {
        return Math.exp(lgamma(x));
    }

    // Zwraca wartość funkcji logarytmu gamma dla x
    private double lgamma(double x) {
        double[] coef = {0.99999999999980993, 676.5203681218851, -1259.1392167224028,
                771.32342877765313, -176.61502916214059, 12.507343278686905,
                -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};
        double sum = coef[0];
        for (int i = 1; i < coef.length; i++) {
            sum += coef[i] / (x + i);
        }
        return Math.log(sum * Math.sqrt(2 * Math.PI)) - (x + 0.5) * Math.log(x + 5.5) + (x + 5.5);
    }

    // Ogranicza wartość do przedziału
    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public void run(int mode) {
        this.mode = mode;
        if(mode != 0){
            SetFunctions setFunctions = new SetFunctions(mode);
            nameFunction = setFunctions.getNameFunction();
            optimum = setFunctions.getOptimum();
        }

        CuckooSearch cuckooSearch = new CuckooSearch(populationSize, probability, alpha, lowerBorder, upperBorder, maxIterations);
        double[] solution = cuckooSearch.run();

        bestSolution = Arrays.toString(solution);
        fitness = cuckooSearch.fitness(solution);
    }
}
