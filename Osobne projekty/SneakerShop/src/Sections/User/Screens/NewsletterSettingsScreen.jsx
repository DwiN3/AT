import { View, Text, TouchableOpacity } from 'react-native';
import React, { useState } from 'react';
import TopNavigationAccountSettings from '../../Navigation/Panels/Top/TopNavigationAccountSettings';
import { styles } from '../Styles/ChangeInformationStyle';
import Animated, { FadeInUp } from 'react-native-reanimated';


export default function NewsletterSettingsScreen() {
    const [selectedValue, setSelectedValue] = useState('daily');

  return (
    <View style={styles.body}>
        <TopNavigationAccountSettings />
        <View style={styles.contentNewsletter}>
            <Animated.Text entering={FadeInUp.duration(1000).springify().delay(100)} style={styles.titleNewsletter} >Wybierz opcję newslettera</Animated.Text>
            <Animated.View entering={FadeInUp.duration(1000).springify().delay(500)}>
                <CustomRadioButton label='Codzienne zestawienie promocji' selected={selectedValue === 'daily'}  onSelect={() => setSelectedValue('daily')} />
            </Animated.View>
            <Animated.View entering={FadeInUp.duration(1000).springify().delay(900)}>
                <CustomRadioButton label='Cotygodniowy przegląd promocji' selected={selectedValue === 'weekly'} onSelect={() => setSelectedValue('weekly')} />
            </Animated.View>
            <Animated.View entering={FadeInUp.duration(1000).springify().delay(1300)}>
                <CustomRadioButton label='Zrezygnuj z otrzymywania newslettera' selected={selectedValue === 'never'} onSelect={() => setSelectedValue('never')} />
            </Animated.View>
        </View>
    </View>
  )
}

const CustomRadioButton = ({label, selected, onSelect}) => (
    <TouchableOpacity onPress={onSelect} style={[styles.customRadioButton, selected ? styles.customRadioButtonSelected : styles.customRadioButtonNormal]} >
        <Text style={selected? styles.customRadioButtonTextSelected : styles.customRadioButtonText}>
            {label}
        </Text>
    </TouchableOpacity>

);