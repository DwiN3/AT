"use client"

import { useEffect, useState } from "react";
import Image from 'next/image';
import { useSearchParams } from "next/navigation";

import { showToast } from "@/lib/showToast";
import MovieCard from '@/components/movie-card/movieCard';
import FilterCarousel from "@/components/movies/genresCarousel";

const MOVIES_URL = "api/movies";


export default function Home() {
  const [movies, setMovies] = useState<Movie[]>([]);
  const searchParams = useSearchParams();
  const filter = searchParams.get("filter");
  const [activeFilter, setActiveFilter] = useState<string>(filter ? filter : "Wszystkie");

  useEffect(() => {
    const fetchMovies = async () => {
      try {
        const response = await fetch(MOVIES_URL, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
  
        if (!response.ok) {
          throw new Error("Failed to fetch movies.");
        }
  
        const data = await response.json();
  
        setMovies(data);
      } catch {
        showToast("Nie udało się pobrać filmów.", true);
  
        return null;
      }
    }

    fetchMovies();
  }, []);

  const filteredMovies = activeFilter === "Wszystkie"
    ? movies
    : movies.filter((movie) =>
        movie.genre.some((genre) => genre.name === activeFilter)
  );

  return (
    <div>
      <div className="relative h-[40svh] p-[4rem] w-full rounded-b-2xl overflow-hidden flex items-end justify-center">
        <Image
          fill
          priority
          alt="Slider image"
          className="absolute inset-0 object-cover w-full h-full"
          src="/images/movie_database.jpg"
        />

        <div className="absolute inset-0 bg-black opacity-70 flex flex-col" />
          <div className='flex flex-col gap-2'>
            <h1 className="relative z-10 text-primary md:text-6xl text-3xl font-semibold text-center italic">
              Baza filmów, która Cię zaskoczy
            </h1>
            <h2  className="relative z-10 text-white md:text-2xl text-lg font-light text-center italic">
              Najciekawsze tytuły czekają na Ciebie!
            </h2>
          </div>
      </div>

      <FilterCarousel activeFilter={activeFilter} onFilterChange={setActiveFilter} />

      {/* <section className="py-8 px-8 max-w-[1640px] m-auto flex flex-row flex-wrap justify-center gap-8">
        {filteredMovies.length > 0 ? (
          filteredMovies.map((movie) => <MovieCard key={movie.id} movie={movie} />)
        ) : (
          <p className="text-white text-xl">Brak filmów pasujących do wybranego filtru.</p>
        )}
      </section> */}

      <section className='py-8 px-8 max-w-[1640px] m-auto flex flex-row flex-wrap justify-center gap-8'>
        {Array.from({ length: 5 }).map((_, index) => (
          filteredMovies.map((movie) => (
            <MovieCard key={`${movie.id}-${index}`} movie={movie} />
          ))
        ))}
      </section>
    </div>
  );
}