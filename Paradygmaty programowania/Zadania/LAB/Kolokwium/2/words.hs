wyrazy :: String -> [String]
wyrazy zdanie = pomoc zdanie "" []

pomoc :: String -> String -> [String] -> [String]
pomoc "" "" akumulator = reverse akumulator
pomoc "" wyraz akumulator = reverse (wyraz:akumulator)
pomoc (c:cs) wyraz akumulator
  | c == ' '  = pomoc cs "" (wyraz:akumulator)
  | otherwise = pomoc cs (wyraz ++ [c]) akumulator

main :: IO ()
main = do
  let zdanie = "To jest przykładowe zdanie"
  let listaWyrazow = wyrazy zdanie
  print listaWyrazow