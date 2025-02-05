"use client"

import { useEffect, useState } from "react";
import Image from 'next/image'

import { showToast } from "@/lib/showToast";
import { useAuth } from "@/providers/authProvider";
import Carousel from "@/components/carousel/carousel";

const MOVIES_URL = "api/movies";
const SHOWINGS_URL = "api/showings"
const RESERVATIONS_URL = "api/reservations"


export default function Home() {
  const [movies, setMovies] = useState<Movie[]>([]);
  const [showings, setShowings] = useState<Showing[]>([]);
  const [reservations, setReservations] = useState<Reservation[]>([]);

  const auth = useAuth();

  useEffect(() => {
    const fetchLatestMovies = async () => {
      try {
        const response = await fetch(`${MOVIES_URL}?limit=10`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
  
        if (response.status === 401) {
          auth.loginRequired();
  
          return null;
        }
  
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

    const fetchShowings = async () => {
      try {
        const response = await fetch(`${SHOWINGS_URL}?limit=10`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
  
        if (response.status === 401) {
          auth.loginRequired();
  
          return null;
        }
  
        if (!response.ok) {
          throw new Error("Failed to fetch movies.");
        }
  
        const data = await response.json();
  
        setShowings(data);
      } catch {
        showToast("Nie udało się pobrać seansów.", true);
  
        return null;
      }
    }

    const fetchReservations = async () => {
      try {
        const response = await fetch(`${RESERVATIONS_URL}/user?limit=10`, {
          method: "GET",
          headers: { "Content-Type": "application/json" },
        });
  
        if (response.status === 401) return null;
        
        if (!response.ok) {
          throw new Error("Failed to fetch movies.");
        }
  
        const data = await response.json();
  
        setReservations(data);
      } catch {
        showToast("Nie udało się pobrać seansów.", true);
  
        return null;
      }
    }

    fetchReservations();
    fetchShowings();
    fetchLatestMovies();
  }, []);

  return (
    <div>
      <div className="relative sm:h-[45svh] h-[25svh] sm:p-[4rem] p-[2rem] w-full rounded-b-2xl overflow-hidden flex items-end justify-center">
        <Image
          fill
          priority
          alt="Slider image"
          className="absolute inset-0 object-cover w-full h-full"
          src="/images/cinema.jpg"
        />

        <div className="absolute inset-0 bg-black opacity-50 flex flex-col" />
          <div className='flex flex-col gap-2'>
          <h1 className="relative z-10 text-white md:text-6xl text-3xl font-semibold text-center italic">
            Witaj w <span className='text-primary font-semibold'>UltimateScreeenSeats</span>
          </h1>
          <h2  className="relative z-10 text-white md:text-2xl text-lg font-light text-center italic">
            Rezerwuj wybrane miejsca na wymarzone filmy.
          </h2>
          </div>
      </div>

      <section className="max-w-[1640px] mx-auto">
        {auth.isAuthenticated &&
          <Carousel reservations={reservations} resource="reservations" title="Nadchodzące rezerwacje" />
        }
        <Carousel resource="showings" showings={showings} title="Nadchodzące seanse" />
        <Carousel movies={movies} resource="movies" title="Najnowsze filmy" />
      </section>
    </div>
  );
}
