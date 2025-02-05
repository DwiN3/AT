"use client"

import { Card, CardBody, CardFooter } from "@nextui-org/card";
import {Image} from '@nextui-org/image';
import { Clock } from "lucide-react";
import { useRouter } from "next/navigation";

import AgeClassificationIcon from "./ageClassification";


interface MovieCardProps {
    movie: Movie
}

export default function MovieCard({ movie }: MovieCardProps) {

    const router = useRouter();

    const handleGoToMovieDetails =(id: number) => {
      router.push(`/movies/${id}`)
    }

    return (
        <Card key={movie.id} isPressable className='w-[20rem] p-[0.25rem] hover:scale-105' shadow="lg" onPress={() => handleGoToMovieDetails(movie.id)}>
          <CardBody className="relative overflow-hidden w-[100%]"> 
            <Image
              alt={movie.title}
              className="object-cover "
              src={movie.image}
            />
          </CardBody>
          <AgeClassificationIcon age={movie.age_classification} />
          <CardFooter className="flex-col gap-3 content-between pt-1">
            <h2 className="text-primary font-semibold text-xl">{movie.title}</h2>
            <p className="text-default-500 text-small flex items-center gap-1"><Clock size={18} /> Time:<span className="font-bold">{movie.movie_length}</span> min</p>
          </CardFooter>
        </Card>
    )
}