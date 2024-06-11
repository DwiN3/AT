import { View, TouchableOpacity, Animated } from 'react-native'
import React, { useEffect, useRef } from 'react'
import Svg, { G, Circle } from 'react-native-svg';
import { styles } from './Styles/onboardingStyles';
import AntDesign from 'react-native-vector-icons/AntDesign';

export const NextButton = ({ percentage, scrollTo }) => {
    const size = 128;
    const strokeWidth = 2;
    const center = size / 2;
    const radius = size / 2 - strokeWidth / 2;
    const circumference = 2 * Math.PI * radius;

    const progressAnimation = useRef(new Animated.Value(0)).current;
    const progressRef = useRef(null);
    const percentageRef = useRef(0);

    const animation = (toValue) => {
        return Animated.timing(progressAnimation, {
            toValue,
            duration: 250,
            useNativeDriver: true
        }).start();
    };

    useEffect(() => {
        percentageRef.current = percentage;
        animation(percentage);
    }, [percentage]);

    useEffect(() => {
        const strokeDashoffset = circumference - (circumference * percentageRef.current) / 100;

        if (progressRef?.current) {
            progressRef.current.setNativeProps({
                strokeDashoffset,
            });
        }
    }, [percentageRef.current]);


  return (
    <View style={styles.container}>
      <Svg width={size} height={size}>
        <G rotation="-90" origin={center}>
            <Circle stroke='#E6E7E8' cx={center} cy={center} r={radius} strokeWidth={strokeWidth} 
            fill='transparent'
            />
            <Circle 
                ref={progressRef}
                stroke='#490e75'
                cx={center}
                cy={center}
                r={radius}
                strokeWidth={strokeWidth}
                strokeDasharray={circumference}
                fill='transparent'
            /> 
        </G>
      </Svg>
      <TouchableOpacity onPress={scrollTo} style={styles.buttonNext} activeOpacity={0.6}>
        <AntDesign name='arrowright' size={32} color='#fff' />
      </TouchableOpacity>
    </View>
  )
}