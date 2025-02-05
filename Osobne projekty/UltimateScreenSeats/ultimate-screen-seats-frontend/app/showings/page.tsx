"use client";

import { useEffect, useState } from "react";
import { DateRangePicker } from "@nextui-org/date-picker";
import { Button } from "@nextui-org/button";
import { RangeValue } from "@react-types/shared";
import { DateValue } from "@internationalized/date";
import Image from "next/image";

import { showToast } from "@/lib/showToast";
import ShowingListCard from "@/components/showing-card/showingListCard";

const SHOWINGS_URL = "/api/showings/list";


export default function Showings() {
    const [showings, setShowings] = useState<ShowingList[]>([]);
    const [localDateRange, setLocalDateRange] = useState<RangeValue<DateValue> | null>(null);
    const [dateRange, setDateRange] = useState<{ startDate: string | null; endDate: string | null }>({
        startDate: null,
        endDate: null,
    });

    const fetchShowings = async () => {
        try {
            const params = new URLSearchParams();

            if (dateRange.startDate) params.append("start_date", dateRange.startDate);
            if (dateRange.endDate) params.append("end_date", dateRange.endDate);

            const response = await fetch(`${SHOWINGS_URL}?${params.toString()}`, {
                method: "GET",
                headers: { "Content-Type": "application/json" },
            });

            if (!response.ok) {
                throw new Error("Failed to fetch showings.");
            }

            const data = await response.json();

            setShowings(data);
        } catch {
            showToast("Nie udało się pobrać seansów.", true);
        }
    };

    useEffect(() => {
        fetchShowings();
    }, []);

    useEffect(() => {
        fetchShowings();
    }, [dateRange]);

    const handleDateChange = (value: RangeValue<DateValue> | null) => {
        setLocalDateRange(value);

        if (value) {
            const startDate = value.start?.toDate("CET")?.toISOString() || null;
            const endDate = value.end?.toDate("CET")?.toISOString() || null;

            setDateRange({ startDate, endDate });
        } else {
            setDateRange({ startDate: null, endDate: null });
        }
    };

    const handleClearDates = () => {
        setLocalDateRange(null);
        setDateRange({ startDate: null, endDate: null });
    };

    return (
        <div>
            <div className="relative h-[40svh] p-[4rem] w-full rounded-b-2xl overflow-hidden flex items-end justify-center">
                <Image
                    fill
                    priority
                    alt="Slider image"
                    className="absolute inset-0 object-cover w-full h-full"
                    src="/images/person_in_cinema.jpg"
                    style={{ objectPosition: 'center 60%' }}
                />
                <div className="absolute inset-0 bg-black opacity-70 flex flex-col" />
                <div className="flex flex-col gap-2">
                    <h1 className="relative z-10 text-primary md:text-6xl text-3xl font-semibold text-center italic">
                        Zobacz, co gramy
                    </h1>
                    <h2 className="relative z-10 text-white md:text-2xl text-lg font-light text-center italic">
                        Twój kolejny filmowy hit czeka na Ciebie!
                    </h2>
                </div>
            </div>

            <div className="py-5 px-8 max-w-[1640px] mx-auto mt-4 flex flex-wrap items-center justify-center gap-8">
                <DateRangePicker
                    className="w-full max-w-sm"
                    label="Zakres dat"
                    selectorButtonPlacement="end"
                    value={localDateRange}
                    onChange={handleDateChange}
                />
                <Button color="primary" size="lg" variant="ghost" onClick={handleClearDates}>
                    Wyczyść daty
                </Button>
            </div>

            <section className="py-8 px-8 max-w-[1640px] m-auto flex flex-row flex-wrap justify-center gap-8">
                {Array.from({ length: 5 }).map((_, index) => (
                    showings.map((showing) => (
                        <ShowingListCard key={`${showing.id}-${index}`} showing={showing} />
                    ))
                ))}
            </section>
        </div>
    );
}