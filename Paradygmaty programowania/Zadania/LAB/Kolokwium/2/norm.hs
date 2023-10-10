
norm :: [Double] -> [Double]
norm xs = map (/ sqrtSum) xs
  where
    sqrtSum = sqrt $ foldl (\acc x -> acc + x * x) 0 xs

main :: IO ()
main = do
  let nums = [1, 2, 3, 4, 5]
  let normalized = norm nums
  print normalized