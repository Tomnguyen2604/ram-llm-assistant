#!/usr/bin/env python3
"""
Quick test of Odin's natural medicine expertise - top 20 questions
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from test import OdinAssistant

def test_natural_medicine_quick():
    """Quick test of Odin's natural medicine responses"""
    print("🌿 Quick Natural Medicine Expertise Test...")
    print("=" * 50)
    
    # Top 20 most important natural medicine questions
    key_questions = [
        # Essential herbs
        "Tell me about turmeric and its benefits",
        "What are the health benefits of ginger?",
        "How does ashwagandha help with stress?",
        "What is echinacea good for?",
        "Tell me about chamomile tea benefits",
        
        # Core concepts
        "What is natural medicine?",
        "How does holistic healing work?",
        "What are adaptogens and how do they work?",
        "What is Ayurveda and how does it work?",
        "Tell me about Traditional Chinese Medicine",
        
        # Common health concerns
        "Natural remedies for anxiety and stress",
        "What herbs help with insomnia?",
        "Natural ways to boost immune system",
        "Herbs for digestive problems",
        "Natural ways to increase energy",
        
        # Practical guidance
        "How do I choose quality herbal supplements?",
        "What should I know about herb-drug interactions?",
        "How do I find a qualified naturopathic doctor?",
        "What should I tell my doctor about natural remedies?",
        "How do I start incorporating natural medicine?"
    ]
    
    try:
        print("Initializing Odin (CPU mode)...")
        odin = OdinAssistant(finetuned_dir="./odin-finetuned", force_cpu=True)
        
        print(f"\n🎯 Testing {len(key_questions)} key natural medicine questions...")
        print("=" * 50)
        
        excellent_responses = 0
        good_responses = 0
        total_questions = len(key_questions)
        
        for i, question in enumerate(key_questions, 1):
            print(f"\n💬 Question {i}: {question}")
            print("-" * 40)
            
            try:
                response = odin.generate_response(question, response_format="text")
                print(f"🤖 Odin: {response}")
                
                # Quick quality assessment
                response_lower = response.lower()
                question_lower = question.lower()
                
                # Check for natural medicine keywords
                natural_keywords = ['natural', 'herb', 'plant', 'healing', 'holistic', 'wellness', 'remedy']
                has_natural_focus = any(keyword in response_lower for keyword in natural_keywords)
                
                # Check for specific topic relevance
                is_topic_relevant = False
                if 'turmeric' in question_lower and 'turmeric' in response_lower:
                    is_topic_relevant = True
                elif 'ginger' in question_lower and 'ginger' in response_lower:
                    is_topic_relevant = True
                elif 'ashwagandha' in question_lower and 'ashwagandha' in response_lower:
                    is_topic_relevant = True
                elif 'stress' in question_lower and ('stress' in response_lower or 'calm' in response_lower):
                    is_topic_relevant = True
                elif 'natural medicine' in question_lower and 'medicine' in response_lower:
                    is_topic_relevant = True
                elif 'holistic' in question_lower and ('holistic' in response_lower or 'whole' in response_lower):
                    is_topic_relevant = True
                
                # Check response length and engagement
                word_count = len(response.split())
                has_follow_up = '?' in response
                is_substantial = word_count >= 20
                
                # Rate the response
                if (has_natural_focus and is_topic_relevant and is_substantial):
                    print("🌟 EXCELLENT - Natural focus, topic-relevant, substantial!")
                    excellent_responses += 1
                elif (has_natural_focus and (is_topic_relevant or is_substantial)):
                    print("✅ GOOD - Natural focus with good content!")
                    good_responses += 1
                elif has_natural_focus:
                    print("✅ GOOD - Natural focus maintained!")
                    good_responses += 1
                else:
                    print("⚠️  Needs improvement - lacks natural medicine focus")
                
            except Exception as e:
                print(f"❌ Error: {e}")
        
        # Results summary
        total_successful = excellent_responses + good_responses
        success_rate = (total_successful / total_questions) * 100
        
        print("\n" + "=" * 50)
        print("🎯 QUICK TEST RESULTS")
        print("=" * 50)
        print(f"🌟 Excellent Responses: {excellent_responses}/{total_questions}")
        print(f"✅ Good Responses: {good_responses}/{total_questions}")
        print(f"📊 Overall Success Rate: {success_rate:.1f}%")
        
        if success_rate >= 85:
            print("\n🎉 OUTSTANDING! Odin is an expert in natural medicine!")
        elif success_rate >= 70:
            print("\n🌟 EXCELLENT! Odin shows strong natural medicine knowledge!")
        elif success_rate >= 50:
            print("\n✅ GOOD! Odin demonstrates solid natural health understanding!")
        else:
            print("\n⚠️  Odin needs more natural medicine training")
        
        print(f"\n🚀 Ready for natural health conversations!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_natural_medicine_quick()