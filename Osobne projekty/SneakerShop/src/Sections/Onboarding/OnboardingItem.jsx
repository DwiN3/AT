import { View, Text, Image } from 'react-native';
import React from 'react';
import { styles } from './Styles/onboardingStyles';
import Animated, { FadeInUp } from 'react-native-reanimated';

export const OnboardingItem = ({item}) => {
  return (
    <View style ={styles.container}>
      <Animated.Image entering={FadeInUp.delay(400).duration(1000).springify().damping(3)} source={item.image} style={styles.image} />
      <View style={styles.textBox}>
          <Animated.Text entering={FadeInUp.duration(1000).springify()} style={styles.title}>{item.title}</Animated.Text>
          <Animated.Text entering={FadeInUp.duration(1000).springify()} style={styles.description}>{item.description}</Animated.Text>
      </View>
    </View>
  );
};