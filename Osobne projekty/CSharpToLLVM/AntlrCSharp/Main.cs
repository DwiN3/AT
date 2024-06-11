using Antlr4.Runtime;
using AntlrCSharp;
using System.IO;
using System.Text;

string inputDirectory = @"C:\Users\dwini\Desktop\CSharpToLLVM\AntlrCSharp\Input";
string outputDirectory = @"C:\Users\dwini\Desktop\CSharpToLLVM\AntlrCSharp\Output";
string[] fileNames = { "HelloWorld.cs", "Functions.cs", "Loops.cs" };

Console.WriteLine("Podaj input:");
Console.WriteLine("1. Hello World");
Console.WriteLine("2. Functions");
Console.WriteLine("3. Loops");
string input = Console.ReadLine();

int choice;
if (!int.TryParse(input, out choice) || choice < 1 || choice > fileNames.Length)
{
    Console.WriteLine("Nieprawidłowy wybór. Wybieram domyślny plik.");
    choice = 1;
}

string selectedFileName = fileNames[choice - 1];
string selectedFilePath = Path.Combine(inputDirectory, selectedFileName);

// Sprawdzenie czy plik istnieje
if (!File.Exists(selectedFilePath))
{
    Console.WriteLine("Wybrany plik nie istnieje. Wybieram domyślny plik.");
    selectedFilePath = Path.Combine(inputDirectory, fileNames[0]);
}

string sourceCode = File.ReadAllText(selectedFilePath);

// Tworzymy obiekt analizatora i leksera
var lexer = new CSharpLexer(new AntlrInputStream(sourceCode));
var tokens = new CommonTokenStream(lexer);
var parser = new CSharpParser(tokens);

// Pobieramy korzeń drzewa składniowego
var tree = parser.compilationUnit();

// Przetwarzanie drzewa składniowego w celu wygenerowania kodu LLVM IR
var llvmGenerator = new LLVMGenerator();
string llvmIRCode = llvmGenerator.Generate(tree);

// Zapisz kod LLVM IR do pliku .ll
string outputFilePath = Path.Combine(outputDirectory, Path.GetFileNameWithoutExtension(selectedFileName) + ".ll");
File.WriteAllText(outputFilePath, llvmIRCode);
Console.WriteLine($"Kod LLVM IR zapisano do: {outputFilePath}");

// Tworzenie pliku .bat do kompilacji kodu LLVM IR
string batFilePath = Path.Combine(outputDirectory, "compile.bat");
string batFileContent = $@"
@echo off
clang {Path.GetFileName(outputFilePath)} -o {Path.GetFileNameWithoutExtension(selectedFileName)}.exe -Wl,-e,main
pause
";
File.WriteAllText(batFilePath, batFileContent);
Console.WriteLine($"Plik .bat zapisano do: {batFilePath}");

// Wyświetlenie drzewa
Console.WriteLine("\nDrzewo Parsowania:");
Console.WriteLine(tree.ToStringTree(parser));