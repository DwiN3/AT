import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, ScrollView, ToastAndroid, KeyboardAvoidingView } from 'react-native';
import { styles } from '../Styles/ChangeInformationStyle';
import TopNavigationAccountSettings from '../../Navigation/Panels/Top/TopNavigationAccountSettings';
import AccountSettingInput from '../Models/AccountSettingInput';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';

export default function ChangeDeliveryInformationScreen() {
    const [delivery, setDelivery] = useState({
        street: 'Konwaliowa',
        streetSecondLine: '25A',
        city: 'Tarnów',
        postalCode: '33-100',
        country: 'Poland'
    });

    const [errorMessage, setErrorMessage] = useState ({
        street: '',
        streetSecondLine: '',
        city: '',
        postalCode: '',
        country: ''
    });
    
    const validate = async () => {
        let errorMessages = {...errorMessage};

        if (delivery.street.trim() === '') {
            errorMessages.street = 'Street address is required!';
        } else {
            errorMessages.street = '';
        }
        if (delivery.streetSecondLine.trim() === '') {
            errorMessages.streetSecondLine = 'Street address is required!';
        } else {
            errorMessages.streetSecondLine = '';
        }
        if (delivery.city.trim() === '') {
            errorMessages.city = 'City is required!';
        } else {
            errorMessages.city = '';
        }
        if (delivery.postalCode.trim() === '') {
            errorMessages.postalCode = 'Postal code is required!';
        } else if (!isValidPostalCode(delivery.postalCode)) {
            errorMessages.postalCode = 'Invalid postal code!';
        } else {
            errorMessages.postalCode = '';
        }
        if (delivery.country.trim() === '') {
            errorMessages.country = 'Country is required!';
        } else {
            errorMessages.country = '';
        }

        setErrorMessage(errorMessages);
        return errorMessages;
    };

    const isValidPostalCode = (postalCode) => {
        const postalCodeRegex = /^[0-9]{2}-?[0-9]{3}$/;
    
        return postalCodeRegex.test(postalCode);
    };

    const handleUpdate = async () => {
        console.log(delivery);   /* DO USUNIĘCIA */
       
        const errorMessages = await validate();

        if (Object.values(errorMessages).some(value => value !== '')) {
            ToastAndroid.showWithGravity('Wrong Input Data!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            return;
        }

        /* TO DO UPDATE */
    }

    const handleChange = (name, value) => {
        /* Postal code check */
        if (name === 'postalCode' && !/^(\d{0,2}-?\d{0,3})?$/.test(value)) {
            return;
        }

        setDelivery(prevAccount => ({
            ...prevAccount,
            [name]: value
        }));
    };

    return (
        <View style={styles.body}>
            <TopNavigationAccountSettings />
            <KeyboardAvoidingView behavior='height' style={styles.form} keyboardVerticalOffset={20}>
            <Animated.ScrollView entering={FadeInUp.duration(500).springify().delay(200)} contentContainerStyle={styles.card}>
                    <View style={styles.header}>
                        <View style={styles.headerTextBox}>
                            <Text style={styles.title}>Edit Delivery Information</Text>
                        </View>
                        <Image style={styles.stamp} source={require('../../../../assets/Images/User/stamp.png')}/>
                    </View>
                    <View>
                        <AccountSettingInput text='Street' value={delivery.street} handleOnChange={(value) => handleChange('street', value)} errorMessage={errorMessage.street} />
                        <AccountSettingInput text='City' value={delivery.city} handleOnChange={(value) => handleChange('city', value)} errorMessage={errorMessage.city} />
                        <AccountSettingInput text='Postal' value={delivery.postalCode} handleOnChange={(value) => handleChange('postalCode', value)} errorMessage={errorMessage.postalCode} />
                        <AccountSettingInput text='Country' value={delivery.country} handleOnChange={(value) => handleChange('country', value)} errorMessage={errorMessage.country} />
                        <TouchableOpacity style={styles.button} onPress={() => handleUpdate()}>
                            <Text style={styles.title}>Update</Text>
                        </TouchableOpacity>
                    </View>
                </Animated.ScrollView>
            </KeyboardAvoidingView>
        </View>
    );
}