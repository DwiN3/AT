"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { Button } from "@nextui-org/button";
import Image from 'next/image';

export default function NotFoundPage() {
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });
    
    const router = useRouter();

    const handleMouseMove = (event: React.MouseEvent) => {
        const { clientX, clientY, currentTarget } = event;
        const { offsetWidth, offsetHeight } = currentTarget as HTMLDivElement;

        const x = Math.min(Math.max(clientX, 64), offsetWidth - 64);
        const y = Math.min(Math.max(clientY, 64), offsetHeight - 64);

        setMousePosition({ x, y });
    };

    const handleRedirect = () => {
        router.push("/");
    };

    return (
        <div className="absolute inset-0 w-full h-full z-0 overflow-hidden">
            <div className="relative w-full h-full" onMouseMove={handleMouseMove}>
                <div className="absolute inset-0 w-full h-full z-0">
                    <Image
                        fill
                        priority
                        alt="Tło strony not found"
                        className="object-cover w-full h-full"
                        src="/images/question-mark.jpg"
                    />
                    <div className="absolute inset-0 bg-black bg-opacity-40 z-5" />

                    <div
                        className="absolute w-40 h-40 bg-primary-500 bg-opacity-70 blur-3xl rounded-full pointer-events-none"
                        style={{
                            top: mousePosition.y - 64,
                            left: mousePosition.x - 64,
                        }}
                    />
                </div>

                <div className="relative px-8 h-full m-auto max-w-7xl flex flex-col gap-8 justify-center items-center z-10">
                    <h1 className="text-5xl font-bold text-primary mb-4 text-center">Strona nie została znaleziona</h1>
                    <p className="text-lg text-white mb-8 italic text-center">
                        Przepraszamy, ale strona, której szukasz, nie istnieje. Możliwe, że została usunięta lub adres URL jest nieprawidłowy.
                    </p>
                    <Button
                        className="px-12 py-6 text-lg font-semibold hover:scale-110"
                        color="primary"
                        radius="full"
                        variant="shadow"
                        onClick={handleRedirect}
                    >
                        Powrót do strony głównej
                    </Button>

                </div>
            </div>
        </div>
    );
}
