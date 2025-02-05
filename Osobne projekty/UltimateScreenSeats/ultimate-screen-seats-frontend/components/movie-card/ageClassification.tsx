"use client"

import { Tooltip } from "@nextui-org/tooltip";
import { useEffect, useState } from "react";

interface AgeClassificationIconProps {
    age: number;
}

export default function AgeClassificationIcon({ age }: AgeClassificationIconProps) {

    const [iconColor, setIconColor] = useState<'success' | 'warning' | 'danger'>('success');

    useEffect(() => {
        if ( age < 12 ) {
            setIconColor('success');
        } else if ( age <= 16) {
            setIconColor('warning');
        } else {
            setIconColor('danger');
        }

    }, [age])

    return (
        <Tooltip closeDelay={100} color={iconColor} content={`Sugerowany wiek od ${age} lat.`}>
            <div className={`bg-${iconColor}-100 w-10 h-10 rounded-full flex justify-center items-center z-10 absolute top-8 right-8 border-3 border-${iconColor}`}>
                <p className={`text-${iconColor}-600 font-medium`}>+{age}</p>
            </div>
        </Tooltip>
    )
}