"use client"

import { Card, CardBody } from "@nextui-org/card"
import { Armchair, CalendarCheck } from "lucide-react";

interface ReservationCardProps {
    reservation: Reservation
}

export default function ReservationCard({ reservation }: ReservationCardProps) {

    return (
        <Card key={reservation.id} isPressable className='w-[20rem] p-[0.5rem] hover:scale-105' shadow="lg">
          <CardBody className="h-full flex-col gap-4 justify-between pt-1">
            <h2 className="text-primary font-semibold text-xl">{reservation.showing.movie.title}</h2>
            <div className="flex flex-col gap-1">
                <p className="text-default-500 text-small flex items-center gap-1"><CalendarCheck size={20} />:<span className="font-bold text-danger">{new Date(reservation.showing.date).toLocaleString('pl-PL')}</span> </p>
                <p className="text-default-500 text-small flex items-center gap-1"><Armchair size={20} />:<span className="font-bold text-success"> Rząd {reservation.seat_row} miejsce {reservation.seat_column}</span> </p>
            </div>
          </CardBody>
        </Card>
    )
}