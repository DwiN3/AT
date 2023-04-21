# Kamil Dereń 35196
# 20.06.2022r

# ZESTAW I ZADANIE 17
X = c (NA, 3, 14, NA, 33, 17, NA, 41)

print('Wektor przed modyfikacjach:')
print(X)

print('Wektor po modyfikacjach:')
X <- X[!X %in% NA]
print(X)



# ZESTAW II ZADANIE 8
function_test <- function(n)
{
  odp = rep(c(0),n)
  for(i in 1:n)
  {
    wybor = as.integer(runif(1, 1, 5))
    zakres = as.integer(runif(1, 1, 5))
    if(wybor == zakres){
      odp[i] = 1
    }
  }
  return(odp)
}

n = 20
odp_funkcja <- function_test(n)
odp_test <- rbinom(n,size=1,prob=0.2)

poprawne_funkcja = 0
poprawne_test = 0

for(i in 1:n)
{
  if(odp_funkcja[i] == 1){
    poprawne_funkcja = poprawne_funkcja + 1
  }
  if(odp_test[i] == 1){
    poprawne_test = poprawne_test + 1
  }
}

print("Liczba prawidlowych odpowiedzi w skrypcie =")
print(poprawne_funkcja)
print("Liczba prawidlowych odpowiedzi w tescie =")
print(poprawne_test)
print("Srednia w skrypcie =")
print(poprawne_funkcja/n)
print("Srednia w tesie =")
mean(poprawne_test/n)

par(mfrow=c(1, 2))
plot(odp_funkcja)
plot(odp_test)



# ZESTAW III ZADANIE 15
path <- "C:/Users/Kamil/Desktop/moth-trap-experiment.csv"
dane <- read.csv(path)
View(dane)

summary(dane)
print("Srednia ilosc zlapanych ciem do typow polapki:")
aggregate(number.of.moths ~ type.of.lure, data=dane, mean)
print("Srednia ilosc zlapanych ciem do lokalizacji:")
aggregate(number.of.moths ~ location, data=dane, mean)

par(mfrow=c(1, 2))
boxplot(number.of.moths ~ type.of.lure, data=dane)
boxplot(number.of.moths ~ location, data=dane)

par(mfrow=c(1, 1))
boxplot(dane$number.of.moths, main="Number of moths", sub=paste("Outlier rows: ", boxplot.stats(dane$number.of.moths)$out))

model <- aov(number.of.moths ~ type.of.lure, data=dane)
summary(model)
confint(model)

library(e1071)
par(mfrow=c(1, 1))
plot(density(dane$number.of.moths), main="Density Plot: Number of moths", ylab="Frequency", sub=paste("Skewness:", round(e1071::skewness(dane$number.of.moths), 2)))
polygon(density(dane$number.of.moths), col="red")



# ZESTAW IV ZADANIE 8
library(MASS)

anorexia_dane = as.data.frame(anorexia)
przed = anorexia_dane$Prewt
po = anorexia_dane$Postwt

View(anorexia_dane)

roznica = przed - po

hist(roznica)
shapiro.test(roznica)



# ZESTAW V ZADANIE 13
library(faraway)

#a
chimiss_dane <- chmiss
summary(chimiss_dane)
View(chimiss_dane)

#b
for(i in 1:47)
{
  if(is.na(chimiss_dane$race[i]) == TRUE){
    chimiss_dane$race[i] = mean(chimiss_dane$race, na.rm=TRUE)
  }
  
  if(is.na(chimiss_dane$fire[i]) == TRUE){
    chimiss_dane$fire[i] = mean(chimiss_dane$fire, na.rm=TRUE)
  }
  
  if(is.na(chimiss_dane$theft[i]) == TRUE){
    chimiss_dane$theft[i] = mean(chimiss_dane$theft, na.rm=TRUE)
  }
  
  if(is.na(chimiss_dane$age[i]) == TRUE){
    chimiss_dane$age[i] = mean(chimiss_dane$age, na.rm=TRUE)
  }
  
  if(is.na(chimiss_dane$involact[i]) == TRUE){
    chimiss_dane$involact[i] = mean(chimiss_dane$involact, na.rm=TRUE)
  }
  if(is.na(chimiss_dane$income[i]) == TRUE){
    chimiss_dane$income[i] = mean(chimiss_dane$income, na.rm=TRUE)
  }
}

View(chimiss_dane)

#c
linearMod <- lm(involact ~ race + fire + theft + age + income, data=chimiss_dane) 
print(linearMod)
summary(linearMod)

#d
model.null = lm(involact ~ 1, data=chimiss_dane)
model.full = lm(involact ~ race + fire + theft + age + income, data=chimiss_dane)
step(model.null, scope = list(upper=model.full), direction="both", data=chimiss_dane)

model.final = lm(involact ~ fire + race + age, data=chimiss_dane)
summary(model.final)
