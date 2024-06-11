import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, ScrollView, ToastAndroid, KeyboardAvoidingView } from 'react-native';
import { styles } from '../Styles/ChangeInformationStyle';
import TopNavigationAccountSettings from '../../Navigation/Panels/Top/TopNavigationAccountSettings';
import AccountSettingInput from '../Models/AccountSettingInput';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';


export default function ChangePersonalInformationScreen() {
    const [account, setAccount] = useState({
        name: 'Jan',
        surname: 'Kowalski',
        phone: '123456789',
        email: 'jan@o2.com',
    });

    const [errorMessage, setErrorMessage] = useState ({
        name: '',
        surname: '',
        phone: '',
        email: '',
    });
    
    const validate = async () => {
        let errorMessages = {...errorMessage};

        if (account.name.trim() === '') {
            errorMessages.name = 'First name is required!';
        } else {
            errorMessages.name = '';
        }
        if (account.surname.trim() === '') {
            errorMessages.surname = 'Last name is required!';
        } else {
            errorMessages.surname = '';
        }
        if (account.phone.trim() === '') {
            errorMessages.phone = 'Phone number is required!';
        } else if (!isValidPhone(account.phone)) {
            errorMessages.phone = 'Invalid phone number!';
        } else {
            errorMessages.phone = '';
        }
        if (account.email.trim() === '') {
            errorMessages.email = 'Email address is required!';
        } else if (!isValidEmail(account.email)) {
            errorMessages.email = 'Invalid email address!';
        } else if (isEmailUsed(account.email)) {
            errorMessages.email = 'Account with this email address already exist!';
        } else {
            errorMessages.email = '';
        }

        setErrorMessage(errorMessages);
        return errorMessages;
    };

    const isValidPhone = (phone) => {
        const nineDigitsRegex = /^\d{9}$/;
        const plusElevenDigitsRegex = /^\+\d{11}$/;
    
        return nineDigitsRegex.test(phone) || plusElevenDigitsRegex.test(phone) ;
    };

    const isValidEmail = (email) => {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    };

    const isEmailUsed = (email) => {
        /* TO DO CHECK EMAIL IN DATABASE */
        return false;
    }

    const handleUpdate = async () => {
        console.log(account);   /* DO USUNIĘCIA */
       
        const errorMessages = await validate();

        if (Object.values(errorMessages).some(value => value !== '')) {
            ToastAndroid.showWithGravity('Wrong Input Data!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            return;
        }

        /* TO DO UPDATE */
    }

    const handleChange = (name, value) => {
        /* Phone check */
        if ( name === 'phone' && !/^\+?([0-9]*)$/.test(value)) {
            return;
        }

        setAccount(prevAccount => ({
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
                            <Text style={styles.title}>Edit Personal Information</Text>
                        </View>
                        <Image style={styles.stamp} source={require('../../../../assets/Images/User/person.png')}/>
                    </View>
                    <View>
                        <AccountSettingInput text='Name' value={account.name} handleOnChange={(value) => handleChange('name', value)} errorMessage={errorMessage.name}/>
                        <AccountSettingInput text='Surname' value={account.surname} handleOnChange={(value) => handleChange('surname', value)} errorMessage={errorMessage.surname} />
                        <AccountSettingInput text='Phone' value={account.phone} handleOnChange={(value) => handleChange('phone', value)} errorMessage={errorMessage.phone} />
                        <AccountSettingInput text='E-mail' value={account.email} handleOnChange={(value) => handleChange('email', value)} errorMessage={errorMessage.email} />

                        <TouchableOpacity style={styles.button} onPress={() => handleUpdate()}>
                            <Text style={styles.title}>Update</Text>
                        </TouchableOpacity>
                    </View>
                </Animated.ScrollView>
            </KeyboardAvoidingView>
        </View>
    );
}