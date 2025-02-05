"use client"

import { useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { ChevronLeft, ChevronRight } from 'lucide-react';
import useEmblaCarousel from 'embla-carousel-react';

import MovieCard from '../movie-card/movieCard';
import ShowingCard from '../showing-card/showingCard';
import ReservationCard from '../reservation-card/reservationCard';


interface CarouselProps {
    title: string;
    resource: string;
    movies?: Movie[]
    showings?: Showing[]
    reservations?: Reservation[]
}

export default function Carousel({ title, resource, movies, showings, reservations }: CarouselProps) {

    const [selectedIndex, setSelectedIndex] = useState(0);
    const [emblaRef, emblaApi] = useEmblaCarousel({ loop: true });
    const autoplayRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
      if (!emblaApi) return;
  
      const onSelect = () => setSelectedIndex(emblaApi.selectedScrollSnap());
  
      emblaApi.on("select", onSelect);
  
      startAutoplay();
  
      return () => {
        stopAutoplay();
        emblaApi.off("select", onSelect);
      };
    }, [emblaApi]);

    const startAutoplay = () => {
        stopAutoplay();
        autoplayRef.current = setInterval(() => {
          if (emblaApi) emblaApi.scrollNext();
        }, 3000);
      };
    
      const stopAutoplay = () => {
        if (autoplayRef.current) {
          clearInterval(autoplayRef.current);
          autoplayRef.current = null;
        }
      };

    const handleMouseEnter = () => stopAutoplay();
    const handleMouseLeave = () => startAutoplay();

    const scrollPrev = () => {
      stopAutoplay();
      emblaApi?.scrollPrev();
    };
    const scrollNext = () => {
      stopAutoplay();
      emblaApi?.scrollNext();
    };

    return (
      <div className="pt-8">
        <div className="flex md:flex-row gap-2 flex-col items-center justify-center relative pb-3">
          <div className="flex-1" />
          <h2 className="text-2xl text-primary-500 font-semibold text-center">
            {title}
          </h2>
          <div className="flex-1 flex justify-end flex-row items-center">
            <Link
              className="underline text-default-500 hover:text-default-800 hover:scale-105"
              href={`/${resource}`}
              title="Zobacz wszystkie"
            >
              Zobacz wszystkie
            </Link>
            <ChevronRight />
          </div>
        </div>
    
        {movies && movies.length > 5 || showings && showings.length  > 5 || reservations && reservations.length  > 5 ? (
          <div
            className="relative w-full overflow-hidden"
            onMouseEnter={handleMouseEnter}
            onMouseLeave={handleMouseLeave}
          >
            <div className="absolute left-0 top-0 h-full md:w-40 w-10 bg-gradient-to-r from-default-50 to-transparent z-10 pointer-events-none md:visible invisible" />
            <div className="absolute right-0 top-0 h-full md:w-40 w-10 bg-gradient-to-r from-transparent to-default-50 z-10 pointer-events-none md:visible invisible" />
    
            <div ref={emblaRef} className="embla__viewport w-full">
              <div className="embla__container flex items-stretch gap-1">
                {movies ?
                movies.map((movie) => (
                  <div
                    key={movie.id}
                    className="embla__slide sm:px-4 px-1 sm:py-6 py-1 sm:flex-[0_0_20%] flex-[0_0_25%] flex"
                  >
                    <MovieCard key={movie.id} movie={movie} />
                  </div>
                ))
                : showings ? 
                showings.map((showing) => (
                  <div
                    key={showing.id}
                    className="embla__slide sm:px-4 px-1 sm:py-6 py-1 sm:flex-[0_0_20%] flex-[0_0_25%] flex"
                  >
                    <ShowingCard key={showing.id} showing={showing} />
                  </div>
                ))
                : reservations ? 
                reservations.map((reservation) => (
                  <div
                    key={reservation.id}
                    className="embla__slide sm:px-4 px-1 sm:py-6 py-1 sm:flex-[0_0_20%] flex-[0_0_25%] flex"
                  >
                    <ReservationCard key={reservation.id} reservation={reservation} />
                  </div>
                ))
                : null 
              }
              </div>
            </div>
    
            <button
              className="absolute top-1/2 left-4 transform -translate-y-1/2 z-10 p-2 bg-default-700 text-default-50 rounded-full hover:bg-default-500"
              onClick={scrollPrev}
            >
              <ChevronLeft />
            </button>
            <button
              className="absolute top-1/2 right-4 transform -translate-y-1/2 z-10 p-2 bg-default-700 text-default-50 rounded-full hover:bg-default-500"
              onClick={scrollNext}
            >
              <ChevronRight />
            </button>
    
            <div className="flex justify-center gap-2 mt-4">
            {(movies || showings || reservations)?.map((_, index) => (
                <button
                  key={index}
                  className={`w-3 h-3 rounded-full ${
                    index === selectedIndex ? "bg-primary-500" : "bg-gray-400"
                  }`}
                  onClick={() => emblaApi?.scrollTo(index)}
                />
              ))}
            </div>
          </div>
        ) : (
          <div className="py-8 px-8 max-w-[1640px] m-auto flex flex-row flex-wrap justify-center gap-8">
            {(movies || showings || reservations)?.map((item) =>
              resource === "movies" ? (
                <MovieCard key={item.id} movie={item as Movie} />
              ) : resource === "showings" ? (
                <ShowingCard key={item.id} showing={item as Showing} />
              ) : resource === "reservations"? (
                <ReservationCard key={item.id} reservation={item as Reservation} />
              ) : null
            )}
          </div>
        )}
      </div>
    );
}