"use client"

import { Card, CardBody, CardFooter } from "@nextui-org/card";
import {Image} from '@nextui-org/image';
import { CalendarDays, Clock } from "lucide-react";
import { useRouter } from "next/navigation";

import AgeClassificationIcon from "../movie-card/ageClassification";


interface ShowingCardProps {
    showing: ShowingList
}

export default function ShowingListCard({ showing }: ShowingCardProps) {

    const router = useRouter();

    const handleGoToMovieDetails =(id: number) => {
      router.push(`/movies/${id}`)
    }

    return (
        <Card key={showing.id} isPressable className='w-[20rem] p-[0.25rem] hover:scale-105' shadow="lg" onPress={() => handleGoToMovieDetails(showing.movie.id)}>
          <CardBody className="relative overflow-hidden w-[100%]"> 
            <Image
              alt={showing.movie.title}
              className="object-cover "
              src={showing.movie.image}
            />
          </CardBody>
          <AgeClassificationIcon age={showing.movie.age_classification} />
          <CardFooter className="flex-col gap-3 content-between pt-1">
            <h2 className="text-primary font-semibold text-xl">{showing.movie.title}</h2>
            <p className="text-default-500 text-small flex items-center gap-1"><Clock size={18} />Time:<span className="font-bold">{showing.movie.movie_length}</span> min</p>
            <p className="text-default-500 text-small flex items-center gap-3">
                <CalendarDays size={24} />
                <span>
                    W kinach od <span className="font-bold text-primary">{new Date(showing.date_from).toLocaleDateString('pl-PL')}</span> 
                    <br/>do <span className="font-bold text-primary">{new Date(showing.date_to).toLocaleDateString('pl-PL')}</span>
                </span>
            </p>
          </CardFooter>
        </Card>
    )
}