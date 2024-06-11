import React, { useEffect, useState } from 'react';
import { View, Text, Image, TouchableOpacity, ScrollView, Alert, ToastAndroid } from 'react-native';
import axios from 'axios';
import { styles } from '../Styles/OrderApprovalScreenStyle';
import TopNavigationPanel from '../../Navigation/Panels/Top/TopNavigationSortPanel';
import BottomNavigationPanel from '../../Navigation/Panels/Bottom/BottomNavigationWorkerPanel';
import * as Keychain from 'react-native-keychain';
import Ionicons from 'react-native-vector-icons/Ionicons';
import Animated, { FadeInUp } from 'react-native-reanimated';
import LoadingModal from '../../Login/Components/LoadingModal';

export default function OrderApprovalScreen() {
    const [orders, setOrders] = useState([]);
    const [sortOption, setSortOption] = useState('desc');
    const [loading, setLoading] = useState(true);
    const [showSort, setShowSort] = useState(false);

    useEffect(() => {
        fetchOrders();
        
    }, []);

    const fetchOrders = async () => {
        const credentials = await Keychain.getGenericPassword();
        if (!credentials) {
            throw new Error('No credentials stored');
        }
        const parsedData = JSON.parse(credentials.password);
        const token = parsedData.token;

        try {
            setLoading(true);
            const response = await axios.get('https://sneakers-api.fly.dev/api/order', {
                headers: { 
                    Authorization: `Bearer ${token}` 
                }
            });

            const newOrders = response.data.filter(order => order.status === 'nowe').sort((a, b) => new Date(b.date) - new Date(a.date));
            setOrders(newOrders);
            setLoading(false);
        } catch (error) {
            console.error('Error fetching orders:', error);
            setLoading(false);
        }
    };

    const handleSort = (option) => {
        setSortOption(option);
        const sortedOrders = [...orders].sort((a, b) => {
            if (option === 'asc') {
                return new Date(a.date) - new Date(b.date);
            } else {
                return new Date(b.date) - new Date(a.date);
            }
        });
        setOrders(sortedOrders);
    };

    const toggleSort = () => {
        setShowSort(!showSort);
    };

    return (
        <View style={styles.body}>
            <TopNavigationPanel onPressSort={toggleSort} />
            {showSort && 
                <Animated.View entering={FadeInUp.duration(1000).springify()} style={styles.sortContent}>
                    <TouchableOpacity style={[styles.sortButton, sortOption === 'asc' && styles.sortButtonChoosen]} onPress={() => handleSort('asc')}>
                        <Animated.Text entering={FadeInUp.duration(1000).springify().delay(100)} style={[styles.sortButtonText, sortOption === 'asc' && styles.sortButtonTextChoosen]}>Sort by Date Ascending</Animated.Text>
                    </TouchableOpacity>
                    <TouchableOpacity style={[styles.sortButton, sortOption === 'desc' && styles.sortButtonChoosen]} onPress={() => handleSort('desc')}>
                        <Animated.Text entering={FadeInUp.duration(1000).springify().delay(500)}  style={[styles.sortButtonText, sortOption === 'desc' && styles.sortButtonTextChoosen]}>Sort by Date Descending</Animated.Text>
                    </TouchableOpacity>
                </Animated.View>
            }
            <ScrollView contentContainerStyle={styles.mainContainer}>
                {!loading ? 
                    orders.map((order, index) => (
                        <OrderItem key={index} order={order} fetchOrders={fetchOrders} />
                    )) : 
                <LoadingModal />}
            </ScrollView>
            <BottomNavigationPanel />
        </View>
    );
}

function OrderItem({ order, fetchOrders }) {
    const [loading, setLoading] = useState(false);

    const updateOrderStatus = async (orderId, status) => {
        console.log(orderId);
        const credentials = await Keychain.getGenericPassword();
        if (!credentials) {
            throw new Error('No credentials stored');
        }
        const parsedData = JSON.parse(credentials.password);
        const token = parsedData.token;
        
    
        try {
            setLoading(true);
            const response = await axios.put(`https://sneakers-api.fly.dev/api/order/${orderId}`, { status },
            {
                headers: { 
                    Authorization: `Bearer ${token}` 
                }
            });
            console.log('Order status updated successfully:', response.data);
            ToastAndroid.showWithGravity('Status zamówienia został zmieniony.', ToastAndroid.SHORT, ToastAndroid.CENTER);
            setLoading(false);
            fetchOrders();
        } catch (error) {
            setLoading(false);
            ToastAndroid.showWithGravity('Przy zmienianiu statusu zamówienia wystąpił błąd.', ToastAndroid.SHORT, ToastAndroid.CENTER);
            console.error('Error updating order status:', error);
        }
    };

    return (
        <Animated.View entering={FadeInUp.duration(1000).springify()} style={styles.setContainer}>
            <Animated.Text entering={FadeInUp.duration(1000).springify().delay(100)} style={styles.contactText}><Text style={styles.contactTextLabel}>Order Date:</Text> {new Date(order.date).toUTCString()}</Animated.Text>
            <View style={styles.contactContainer}>
                <Animated.Text entering={FadeInUp.duration(1000).springify().delay(400)} style={styles.contactText}><Text style={styles.contactTextLabel}>Name:</Text> {order.contact.name} {order.contact.lastname}</Animated.Text>
                <Animated.Text entering={FadeInUp.duration(1000).springify().delay(700)} style={styles.contactText}><Text style={styles.contactTextLabel}>Phone:</Text> {order.contact.phoneNumber}</Animated.Text>
                <Animated.Text entering={FadeInUp.duration(1000).springify().delay(1000)} style={styles.contactText}><Text style={styles.contactTextLabel}>Address:</Text> {order.contact.street} {order.contact.streetNumber}, {order.contact.city}, {order.contact.postalCode}, {order.contact.country}</Animated.Text>
            </View>
            <Animated.View entering={FadeInUp.duration(1200).springify()} style={{ marginTop: 25 }}>
                {order.stocks.map((stock, index) => (
                    <CartItem key={index} stock={stock} />
                ))}
            </Animated.View>
            <Animated.View entering={FadeInUp.duration(1000).springify().delay(2300)}>
                <TouchableOpacity style={styles.acceptButton} onPress={() => updateOrderStatus(order.id, 'zrealizowane')}>
                    <Ionicons name={'checkmark-circle-outline'} size={38} color= {'#fff'} style={styles.IconsSize} />
                    <Text style={styles.acceptText}> Shipped</Text>
                </TouchableOpacity>
            </Animated.View>
            <LoadingModal visible={loading}/>
        </Animated.View>
    );
}

function CartItem({ stock }) {
    const productImageUrl = stock.product.photos[0]?.imgUrl || '';

    return (
        <Animated.View entering={FadeInUp.duration(1000).springify().delay(1200)} style={styles.cartProduct}>
            <Animated.View entering={FadeInUp.duration(1000).springify()} style={styles.productImageContainer}>
                <Image style={styles.productImage} source={{ uri: productImageUrl }} />
            </Animated.View>
            <Animated.View entering={FadeInUp.duration(1000).springify().delay(1400)} style={styles.productInfoContainer}>
                <Animated.Text entering={FadeInUp.duration(1000).springify().delay(1600)} style={styles.nameProductText}>{stock.product.producer} {stock.product.model}</Animated.Text>
                <Animated.Text entering={FadeInUp.duration(1000).springify().delay(1800)} style={styles.sizeText}><Text style={styles.contactTextLabel}>Size:</Text> {stock.product.size}</Animated.Text>
                <Animated.Text entering={FadeInUp.duration(1000).springify().delay(2000)} style={styles.quantityText}><Text style={styles.contactTextLabel}>Quantity:</Text> {stock.quantity}</Animated.Text>
            </Animated.View>
        </Animated.View>
    );
}
